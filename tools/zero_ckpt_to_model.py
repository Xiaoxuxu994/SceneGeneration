#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 accelerate + DeepSpeed ZeRO-2 的训练 checkpoint（checkpoint_N/ 下的 mp_rank_00_model_states.pt +
bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt）合并成一个普通的 PyTorch state_dict，可直接
`model.load_state_dict(torch.load(...))`，或作为训练配置的 model.ckpt=<文件>。

两种模式：
  --mode fp32（默认）：用 DeepSpeed 的 zero_to_fp32 从各 rank 的优化器分片里拼出 fp32 主权重（训练时真正更新的那份，最精确）。
                     需要全部 optim_states 分片都在，内存约 分片总大小 + 2×模型 fp32（3B 模型约 50GB）。
  --mode bf16：只读 mp_rank_00_model_states.pt 里的 module（ZeRO-2 每个 rank 都存完整的 bf16 权重），不需要优化器分片，
               快；精度是 bf16（推理 / 评测通常够用，继续训练或做初始化建议 fp32）。

    python tools/zero_ckpt_to_model.py <checkpoint_N 目录> [--out <文件>] [--mode fp32|bf16] [--safetensors]
默认输出 <checkpoint_N>/model_<mode>.pt（--safetensors 时为 .safetensors）。
"""
import argparse
import glob
import os
import os.path as osp
import sys
import time

import torch


def find_model_states(ckpt_dir):
    hits = sorted(glob.glob(osp.join(ckpt_dir, "**", "mp_rank_00_model_states.pt"), recursive=True))
    if not hits:
        sys.exit(f"[ERROR] 在 {ckpt_dir} 下找不到 mp_rank_00_model_states.pt（不是 DeepSpeed checkpoint？）")
    if len(hits) > 1:
        print(f"[WARN] 找到多个 model_states，用第一个：{hits}", flush=True)
    return hits[0]


def strip_prefix(sd):
    for prefix in ("module.", "_orig_mod."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir", help="checkpoint_N 目录（或其下 DeepSpeed 的子目录，如 pytorch_model/）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--safetensors", action="store_true", help="另存为 .safetensors（需要 safetensors 包）")
    args = ap.parse_args()

    t0 = time.time()
    ms_path = find_model_states(args.ckpt_dir)
    ds_dir = osp.dirname(ms_path)                      # DeepSpeed 的 tag 目录（accelerate 下通常是 pytorch_model/）
    n_shards = len(glob.glob(osp.join(ds_dir, "*_optim_states.pt")))
    print(f"[ckpt] model_states: {ms_path}  ({osp.getsize(ms_path) / 1e9:.2f} GB)，optim 分片 {n_shards} 个", flush=True)

    if args.mode == "fp32":
        if n_shards == 0:
            sys.exit("[ERROR] fp32 模式需要 *_optim_states.pt 分片；没有的话用 --mode bf16")
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        sd = get_fp32_state_dict_from_zero_checkpoint(osp.dirname(ds_dir), tag=osp.basename(ds_dir))
    else:
        states = torch.load(ms_path, map_location="cpu", weights_only=False)
        sd = states["module"]
        del states
    sd = strip_prefix(dict(sd))

    n_param = sum(v.numel() for v in sd.values())
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    tops = sorted({k.split(".")[0] for k in sd})
    print(f"[ckpt] {len(sd)} tensors, {n_param / 1e9:.3f} B params, dtypes {dtypes}", flush=True)
    print(f"[ckpt] top-level modules: {tops}", flush=True)

    out = args.out or osp.join(args.ckpt_dir, f"model_{args.mode}.{'safetensors' if args.safetensors else 'pt'}")
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    if args.safetensors:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, out)
    else:
        torch.save(sd, out)
    print(f"[ckpt] saved {out} ({osp.getsize(out) / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
