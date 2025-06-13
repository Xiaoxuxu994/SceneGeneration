from copy import deepcopy
import torch.nn as nn

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register_module(self, module=None, name=None):
        def _register(cls):
            module_name = name or cls.__name__
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self._name}")
            self._module_dict[module_name] = cls
            return cls

        return _register(module) if module else _register

    def get(self, key):
        return self._module_dict.get(key, None)

    def build(self, cfg):
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        module_type = cfg.pop("type")
        module_cls = self.get(module_type)
        if module_cls is None:
            raise KeyError(f"{module_type} is not registered in {self._name}")
        return module_cls(**cfg)


MODELS = Registry("model")
SCHEDULERS = Registry("scheduler")
DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")


def build_module(module, builder, **kwargs):
    if isinstance(module, dict):
        cfg = deepcopy(module)
        cfg.update(kwargs)
        return builder.build(cfg)
    elif isinstance(module, nn.Module):
        return module
    elif module is None:
        return None
    else:
        raise TypeError(f"Unsupported module type: {type(module)}")