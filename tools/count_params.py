# -*- coding: utf-8 -*-
"""
统计 Pi3 各模型配置的参数量（不加载权重；参数建在 meta device 上，几秒出结果，不占显存/内存）。
在仓库根目录运行：
    python tools/count_params.py                                    # 默认对比：原始 Pi3 / 新的 huge+ 小 decoder / 现在的 3B
    python tools/count_params.py dinov3_huge+_1.43B dinov3_huge+_3Bv2
"""
import os
import sys
from collections import OrderedDict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pi3.models.pi3_training as M  # noqa: E402

DEFAULT = ["dinov2_large_0.95B", "dinov3_huge+_1.43B", "dinov3_huge+_3Bv2"]
GROUPS = OrderedDict([
    ("encoder", ("encoder.",)),
    ("decoder", ("decoder.", "project_e2d.", "project_norm.", "register_token")),
    ("point", ("point_decoder.", "point_head.")),
    ("camera", ("camera_decoder.", "camera_head.")),
    ("global/conf", ("global_points_decoder.", "global_point_head.", "conf_")),
])


def _no_pretrained(builder):
    def wrapped(*a, pretrained=True, model_dir=None, **kw):
        return builder(*a, pretrained=False, **kw)
    return wrapped


def build(model_size):
    cfg = M.Pi3._build_backbone_cfg()
    orig = M.Pi3._build_backbone_cfg
    for enc in cfg.values():
        for spec in enc.values():
            spec["builder"] = _no_pretrained(spec["builder"])
    M.Pi3._build_backbone_cfg = staticmethod(lambda: cfg)
    try:
        from accelerate import init_empty_weights   # 只把参数放到 meta，其余张量照常（torch.device("meta") 会让 .item() 报错）
        with init_empty_weights():
            return M.Pi3(model_size=model_size, load_vggt=False, freeze_encoder=False, ckpt=None)
    finally:
        M.Pi3._build_backbone_cfg = staticmethod(orig)


def count(model):
    out = OrderedDict((k, 0) for k in GROUPS)
    out["other"] = 0
    for n, p in model.named_parameters():
        for g, prefixes in GROUPS.items():
            if n.startswith(prefixes):
                out[g] += p.numel()
                break
        else:
            out["other"] += p.numel()
    out["total"] = sum(out.values())
    return out


def main(names):
    cfg = M.Pi3._build_backbone_cfg()
    rows = []
    for name in names:
        enc_type, enc_size, var = M.Pi3._resolve_model_spec(None, name)
        dec = cfg[enc_type][enc_size]["variants"].get(var) or cfg[enc_type][enc_size]["default_decoder"]
        c = count(build(name))
        rows.append((name, dec, c))
    keys = list(rows[0][2].keys())
    print(f"{'model_size':24s} {'decoder (dim/heads/depth/head)':32s} " + " ".join(f"{k:>11s}" for k in keys))
    for name, d, c in rows:
        ds = f"{d['dec_embed_dim']}/{d['dec_num_heads']}/{d['dec_depth']}/{d['head_depth']}"
        print(f"{name:24s} {ds:32s} " + " ".join(f"{c[k] / 1e6:10.1f}M" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
