#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 / 精简一个合并后的模型 checkpoint（zero_to_fp32 的输出目录或文件、.bin / .pt / .safetensors）。

  1) 查看：按顶层模块统计张量数、参数量、dtype、字节数，并找出同名前缀下重复存放的同一份权重
       python tools/slim_ckpt.py <ckpt 路径>
  2) 精简：只保留指定的顶层模块（或去掉某些模块），可转 bf16，存成一个文件
       python tools/slim_ckpt.py <ckpt 路径> --keep encoder decoder ... --out model_slim.safetensors
       python tools/slim_ckpt.py <ckpt 路径> --drop teacher ema --dtype bf16 --out model_bf16.safetensors
  --check-model：用训练配置实例化 Pi3（不加载权重），只保留模型自己 state_dict 里有的 key，并报告缺失 / 多余的 key
       python tools/slim_ckpt.py <ckpt 路径> --check-model --out model_slim.safetensors
"""
import argparse
import glob
import hashlib
import os
import os.path as osp
import sys
from collections import defaultdict

import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_any(path):
    if osp.isdir(path):
        files = sorted(glob.glob(osp.join(path, "*.safetensors"))) or sorted(glob.glob(osp.join(path, "*.bin")))
        if not files:
            sys.exit(f"[ERROR] {path} 下没有 .safetensors / .bin")
    else:
        files = [path]
    sd = {}
    for f in files:
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd.update(load_file(f))
        else:
            obj = torch.load(f, map_location="cpu", weights_only=False)
            sd.update(obj.get("module", obj.get("state_dict", obj)) if isinstance(obj, dict) and not torch.is_tensor(next(iter(obj.values()), None)) else obj)
    print(f"[load] {len(files)} file(s), {sum(osp.getsize(f) for f in files) / 1e9:.2f} GB on disk", flush=True)
    return sd


def summarize(sd, depth):
    stats = defaultdict(lambda: [0, 0, 0, set()])
    for k, v in sd.items():
        key = ".".join(k.split(".")[:depth])
        s = stats[key]
        s[0] += 1; s[1] += v.numel(); s[2] += v.numel() * v.element_size(); s[3].add(str(v.dtype).replace("torch.", ""))
    total = sum(s[1] for s in stats.values())
    print(f"\n{'module':40s} {'tensors':>8s} {'params(M)':>11s} {'GB':>7s}  dtype", flush=True)
    for key, (n, p, b, dt) in sorted(stats.items(), key=lambda x: -x[1][1]):
        print(f"{key:40s} {n:8d} {p / 1e6:11.1f} {b / 1e9:7.2f}  {','.join(sorted(dt))}")
    print(f"{'TOTAL':40s} {len(sd):8d} {total / 1e6:11.1f} {sum(s[2] for s in stats.values()) / 1e9:7.2f}", flush=True)


def find_duplicates(sd):
    """内容完全相同的大张量（≥1M 元素）：共享 / 重复存放的权重。"""
    by_hash = defaultdict(list)
    for k, v in sd.items():
        if v.numel() >= 1_000_000:
            t = v.detach().contiguous().view(-1)
            h = hashlib.md5(t[:: max(1, t.numel() // 4096)].float().numpy().tobytes()).hexdigest()
            by_hash[(tuple(v.shape), str(v.dtype), h)].append(k)
    dups = [ks for ks in by_hash.values() if len(ks) > 1]
    extra = sum(sd[ks[0]].numel() * sd[ks[0]].element_size() * (len(ks) - 1) for ks in dups)
    print(f"\n[dup] {len(dups)} groups of identical large tensors, {extra / 1e9:.2f} GB stored more than once", flush=True)
    for ks in dups[:10]:
        print("   ", ks[:4], "..." if len(ks) > 4 else "")


def model_keys():
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    cfg = OmegaConf.load(osp.join(ROOT, "configs", "model", "pi3.yaml"))
    cfg = cfg.get("model", cfg)
    if "ckpt" in cfg:
        cfg.ckpt = None
    with torch.device("meta"):
        model = instantiate(cfg)
    return set(model.state_dict().keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--depth", type=int, default=1, help="统计时按前几级模块名分组")
    ap.add_argument("--keep", nargs="*", default=None, help="只保留这些顶层模块（key 的第一段）")
    ap.add_argument("--drop", nargs="*", default=None, help="去掉这些顶层模块")
    ap.add_argument("--check-model", action="store_true", help="只保留 Pi3 模型 state_dict 里有的 key（按 configs/model/pi3.yaml 实例化）")
    ap.add_argument("--dtype", choices=["keep", "fp32", "bf16", "fp16"], default="keep")
    ap.add_argument("--out", default=None, help="精简后的输出文件（.safetensors 或 .pt）；不给则只查看")
    args = ap.parse_args()

    sd = load_any(args.path)
    summarize(sd, args.depth)
    find_duplicates(sd)

    if args.keep:
        sd = {k: v for k, v in sd.items() if k.split(".")[0] in set(args.keep)}
    if args.drop:
        sd = {k: v for k, v in sd.items() if k.split(".")[0] not in set(args.drop)}
    if args.check_model:
        mk = model_keys()
        missing, extra = sorted(mk - set(sd)), sorted(set(sd) - mk)
        print(f"\n[model] model has {len(mk)} keys; missing in ckpt {len(missing)} {missing[:5]}; not in model {len(extra)} {extra[:5]}", flush=True)
        sd = {k: v for k, v in sd.items() if k in mk}
    if args.out is None:
        return
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype)
    if dt is not None:
        sd = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in sd.items()}
    sd = {k: v.contiguous().clone() for k, v in sd.items()}      # 去掉共享存储，避免保存时重复或报错
    print("\n[out] after filtering:")
    summarize(sd, args.depth)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    if args.out.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(sd, args.out)
    else:
        torch.save(sd, args.out)
    print(f"[out] saved {args.out} ({osp.getsize(args.out) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
