import torch
from collections import OrderedDict

def get_activation_size(model, input_size, device='cpu'):
    activations = OrderedDict()

    def register_hook(module):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                size = output.numel() * output.element_size()
            elif isinstance(output, (tuple, list)):
                size = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            else:
                size = 0
            activations[module_name] = size / 1024**2  # 转为 MB
        return hook

    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList) and module != model:
            module_name = name
            hooks.append(module.register_forward_hook(register_hook(module)))

    dummy_input = torch.randn(*input_size).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    print("\n🔍 每层激活内存使用情况(单位:MB):")
    for name, size in activations.items():
        print(f"{name:40s} : {size:.2f} MB")

    total = sum(activations.values())
    print(f"\n💡 总激活内存约为：{total:.2f} MB")

def register_activation_hooks(model, rank, logger):
    def hook_fn(module, input, output):
        size = 0
        if isinstance(output, torch.Tensor):
            size = output.numel() * output.element_size()
            dtype = output.dtype
        elif isinstance(output, (tuple, list)):
            size = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            dtype = output[0].dtype
        elif isinstance(output, dict):
            size = sum(v.numel() * v.element_size() for v in output.values() if isinstance(v, torch.Tensor))
            dtype = output[list(output.keys())[0]].dtype
        
        logger.info(f"rank = {rank}, [{module.__class__.__name__}] {dtype} Activation: {size / 1024**2:.2f} MB, Allocated: {torch.npu.memory_allocated() / 1024**2:.2f} MB reserved : {torch.npu.memory_reserved() / 1024**2:.2f} MB")

    hooks = []
    for _, module in model.named_modules():
        if module != model:
            hooks.append(module.register_forward_hook(hook_fn))
    return hooks