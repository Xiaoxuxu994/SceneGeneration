import os
import torch
from safetensors.torch import load_file
from collections import defaultdict


def freeze_selected_parts(pi3, freeze_module_names=None, freeze_param_names=None, verbose=True):
    """
    只冻结明确指定的模块/参数，其他参数保持原状。
    
    Args:
        pi3: Pi3 实例，比如 model.depth_aligner
        freeze_module_names: 要冻结的顶层 module 名字列表
        freeze_param_names: 要冻结的顶层 parameter 名字列表
    """
    freeze_module_names = set(freeze_module_names or [])
    freeze_param_names = set(freeze_param_names or [])

    frozen_names = []

    # 1) 冻结指定 module
    for module_name, module in pi3.named_children():
        if module_name in freeze_module_names:
            for param_name, p in module.named_parameters(recurse=True):
                p.requires_grad = False
                full_name = f"{module_name}.{param_name}" if param_name else module_name
                frozen_names.append(full_name)

    # 2) 冻结指定的“独立参数” （例如 register_token）
    for param_name, p in pi3.named_parameters(recurse=False):
        if param_name in freeze_param_names:
            p.requires_grad = False
            frozen_names.append(param_name)

    if verbose:
        print(f"[ INFO ] Freeze modules: {sorted(freeze_module_names)}")
        print(f"[ INFO ] Freeze params  : {sorted(freeze_param_names)}")
        print(f"[ INFO ] Frozen param count: {len(frozen_names)}")

    return frozen_names

def print_trainable_summary(pi3):
    frozen = []
    trainable = []

    for name, p in pi3.named_parameters():
        if p.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print("\n[ INFO ] ===== Frozen Parameters =====")
    for name in frozen:
        print(name)

    print("\n[ INFO ] ===== Trainable Parameters =====")
    for name in trainable:
        print(name)

    frozen_num = sum(p.numel() for _, p in pi3.named_parameters() if not p.requires_grad)
    trainable_num = sum(p.numel() for _, p in pi3.named_parameters() if p.requires_grad)
    total_num = frozen_num + trainable_num

    print(f"\n[ INFO ] Frozen params   : {frozen_num:,}")
    print(f"[ INFO ] Trainable params: {trainable_num:,}")
    print(f"[ INFO ] Total params    : {total_num:,}")

def assert_frozen_status(pi3):
    # 应该被冻结的前缀
    must_frozen_prefixes = ["encoder.", "decoder."]
    must_frozen_exact = ["register_token"]

    # 不应该被冻结的前缀
    must_train_prefixes = [
        "point_decoder.", "point_head.",
        "xy_decoder.", "xy_head.",
        "depth_decoder.", "depth_head.",
        "conf_decoder.", "conf_head.",
        "camera_decoder.", "camera_head.",
    ]

    for name, p in pi3.named_parameters():
        if any(name.startswith(prefix) for prefix in must_frozen_prefixes) or name in must_frozen_exact:
            assert p.requires_grad is False, f"Should be frozen but trainable: {name}"

        if any(name.startswith(prefix) for prefix in must_train_prefixes):
            assert p.requires_grad is True, f"Should be trainable but frozen: {name}"

    print("[ INFO ] Freeze status check passed.")

def _maybe_strip_module_prefix(state_dict, target_state_dict):
    src_has_module = any(k.startswith("module.") for k in state_dict.keys())
    dst_has_module = any(k.startswith("module.") for k in target_state_dict.keys())

    if src_has_module and not dst_has_module:
        return {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    return dict(state_dict)


def _copy_old_tensor_into_new_shape(old_tensor, new_tensor):
    """
    用旧 tensor 尽量填充到新 tensor 里：
    - 1D: copy 到前面，剩余位置置 0
    - 2D: copy 到左上角，剩余位置置 0
    """
    out = torch.zeros_like(new_tensor)

    if old_tensor.ndim == 1:
        n = min(old_tensor.shape[0], new_tensor.shape[0])
        out[:n] = old_tensor[:n]
    elif old_tensor.ndim == 2:
        n0 = min(old_tensor.shape[0], new_tensor.shape[0])
        n1 = min(old_tensor.shape[1], new_tensor.shape[1])
        out[:n0, :n1] = old_tensor[:n0, :n1]
    else:
        raise NotImplementedError(
            f"Only support 1D/2D tensor expansion, got ndim={old_tensor.ndim}"
        )

    return out


def _adapt_camera_head_more_mlps_for_register(state_dict, target_state_dict, enable=False):
    """
    当旧 ckpt 没有 register token 输入，而新模型 CameraHead 开启了
    input_register_token_num=5 时，尽量复用旧的 more_mlps 权重。

    适配规则：
    - more_mlps.0.weight: old [512, 512] -> new [3072, 3072]，拷到左上角
    - more_mlps.0.bias  : old [512]      -> new [3072]，拷到前面
    - more_mlps.2.weight: old [512, 512] -> new [512, 3072]，拷到左边
    - more_mlps.2.bias  : old [512]      -> new [512]，直接拷贝
    """
    state_dict = _maybe_strip_module_prefix(state_dict, target_state_dict)
    adapted_keys = []

    if not enable:
        return state_dict, adapted_keys

    for k, target_v in target_state_dict.items():
        if "camera_head.more_mlps" not in k:
            continue
        if k not in state_dict:
            continue

        src_v = state_dict[k]
        if tuple(src_v.shape) == tuple(target_v.shape):
            continue

        # 只适配这四个参数，其他 mismatch 的 key 直接删掉，避免 load_state_dict 报 shape 错
        if k.endswith((".0.weight", ".0.bias", ".2.weight", ".2.bias")):
            old_shape = tuple(src_v.shape)
            new_shape = tuple(target_v.shape)
            state_dict[k] = _copy_old_tensor_into_new_shape(src_v, target_v)
            adapted_keys.append(k)
            print(f"[ INFO ] Adapt {k}: {old_shape} -> {new_shape}")
        else:
            print(
                f"[ WARN ] Remove mismatched key: {k}, "
                f"ckpt shape={tuple(src_v.shape)}, target shape={tuple(target_v.shape)}"
            )
            del state_dict[k]

    return state_dict, adapted_keys

def summarize_state_dict_coverage(loaded_state_dict, target_state_dict, title="ckpt coverage"):
    """
    统计:
    1) 总参数覆盖率
    2) 按模块前缀的覆盖率
    3) 完全匹配 / shape不匹配 / 缺失 / 多余 key
    """

    loaded_keys = set(loaded_state_dict.keys())
    target_keys = set(target_state_dict.keys())

    matched_keys = []
    shape_mismatch_keys = []
    missing_keys = []
    unexpected_keys = []

    matched_numel = 0
    total_numel = 0

    module_total = defaultdict(int)
    module_loaded = defaultdict(int)

    for k, v in target_state_dict.items():
        total_numel += v.numel()

        top_prefix = k.split(".")[0]
        module_total[top_prefix] += v.numel()

        if k not in loaded_state_dict:
            missing_keys.append(k)
            continue

        src_v = loaded_state_dict[k]
        if tuple(src_v.shape) == tuple(v.shape):
            matched_keys.append(k)
            matched_numel += v.numel()
            module_loaded[top_prefix] += v.numel()
        else:
            shape_mismatch_keys.append(
                (k, tuple(src_v.shape), tuple(v.shape))
            )

    for k in loaded_state_dict.keys():
        if k not in target_state_dict:
            unexpected_keys.append(k)

    print("\n" + "=" * 80)
    print(f"[ INFO ] {title}")
    print("=" * 80)

    print(f"[ INFO ] target keys          : {len(target_keys)}")
    print(f"[ INFO ] loaded keys          : {len(loaded_keys)}")
    print(f"[ INFO ] matched keys         : {len(matched_keys)}")
    print(f"[ INFO ] missing keys         : {len(missing_keys)}")
    print(f"[ INFO ] unexpected keys      : {len(unexpected_keys)}")
    print(f"[ INFO ] shape mismatch keys  : {len(shape_mismatch_keys)}")

    coverage = matched_numel / total_numel if total_numel > 0 else 0.0
    print(f"[ INFO ] param coverage       : {matched_numel:,} / {total_numel:,} = {coverage:.4%}")

    print("\n[ INFO ] module-wise coverage:")
    for module_name in sorted(module_total.keys()):
        mod_total = module_total[module_name]
        mod_loaded = module_loaded[module_name]
        mod_cov = mod_loaded / mod_total if mod_total > 0 else 0.0
        print(f"  - {module_name:20s}: {mod_loaded:>12,} / {mod_total:<12,} = {mod_cov:.4%}")

    if shape_mismatch_keys:
        print("\n[ WARN ] shape mismatch keys (first 20):")
        for item in shape_mismatch_keys[:20]:
            k, s1, s2 = item
            print(f"  - {k}: ckpt{s1} vs model{s2}")

    if missing_keys:
        print("\n[ WARN ] missing keys (first 20):")
        for k in missing_keys[:20]:
            print(f"  - {k}")

    if unexpected_keys:
        print("\n[ WARN ] unexpected keys (first 20):")
        for k in unexpected_keys[:20]:
            print(f"  - {k}")

    print("=" * 80 + "\n")

    return {
        "coverage": coverage,
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }