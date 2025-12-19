import yaml
import json
import os
from functools import cache
from dotted_dict import DottedDict



class NestedYamlLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)


@cache
def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.load(f, NestedYamlLoader)


def include_constructor(loader, node):
    included_file = loader.construct_scalar(node)
    
    if not os.path.isabs(included_file):
        included_file = os.path.join(loader._root, included_file)
    
    if not os.path.exists(included_file):
        raise FileNotFoundError(f"YAML include file not found: {included_file}")
    
    return load_yaml(included_file)


NestedYamlLoader.add_constructor("!include", include_constructor)


def MergeDicts(default_cfg: DottedDict, custom_cfg: DottedDict) -> DottedDict:
    
    merged_cfg = custom_cfg
    
    for k in default_cfg.keys():
        if custom_cfg.get(k) is not None:
            # Attribute found in default setting, check class consistency for further merge.
            assert default_cfg[k].__class__ == custom_cfg[k].__class__, \
                "The same attribute of custom and default config must be the same class for merge."
            # Merge if the attribute is a dict, otherwise keep the default value unchanged. 
            if isinstance(default_cfg[k], DottedDict) and isinstance(custom_cfg[k], DottedDict):
                merged_cfg[k] = MergeDicts(default_cfg[k], custom_cfg[k])
        else:
            # Key not set in custom setting, use the default value.
            merged_cfg[k] = default_cfg[k]
    
    return merged_cfg


def LoadDefaultConfig(cfg_dict: DottedDict):
    # Recurse to process the sub-configs with default settings.
    processed_cfg = DottedDict()
    for k in cfg_dict.keys():
        if k == "default":
            continue
        if isinstance(cfg_dict[k], DottedDict):
            processed_cfg[k] = LoadDefaultConfig(cfg_dict[k])
        else:
            processed_cfg[k] = cfg_dict[k]
    
    # Default config found, merge unset default properties. 
    if "default" in cfg_dict.keys():
        assert isinstance(cfg_dict.default, DottedDict), "The config attribute 'default' must be a dict."
        for k in processed_cfg.keys():
            assert isinstance(processed_cfg[k], DottedDict), "The config attribute parrallel to 'default' must be a dict."
            processed_cfg[k] = MergeDicts(cfg_dict.default, processed_cfg[k])
    
    return processed_cfg


def LoadConfig(config_path) -> DottedDict:
    with open(config_path, "r+") as f:
        if config_path.endswith(".yaml"):
            cfg = yaml.load(f, Loader=NestedYamlLoader)
            cfg = LoadDefaultConfig(DottedDict(cfg))
        elif config_path.endswith(".json"):
            cfg = json.load(f)
            cfg = DottedDict(cfg)
        else:
            raise NotImplementedError(f"Configuration file type '{config_path.split('.')[-1]}' is not implemented.")
    return cfg


__all__ = [LoadConfig]