# -*- coding: utf-8 -*-
"""
单个数据集的调试入口：每个 loader 文件的 `if __name__ == "__main__":` 都调用 debug_main(该类)，所以每个脚本都能单独启动验证。

在仓库根目录运行（任选一种）：
    python -m datasets.dl3dv_dataset --n 3
    PYTHONPATH=. python datasets/dl3dv_dataset.py --n 3

参数默认从 datamix 配置里取该数据集的块（按 _target_ 的类名自动匹配），命令行可以覆盖：
    --config configs/data/datamix_all_v9-5.yaml   配置文件
    --name DL3DV                                  配置里的键名（同一个类出现多次时指定）
    --data_root obs://bucket/xxx                  覆盖 data_root（本地或 obs://）
    --cache_dir obs://bucket/dataset_cache        覆盖缓存目录；--cache_file 直接指定缓存 pkl
    --remap /efs-m3ga/xxx=obs://bucket/xxx        路径前缀映射，可重复
    --set key=value                               覆盖任意构造参数（值按 YAML 解析），可重复
    --n 3 / --idx 0 10 20                         取哪些样本
    --res 512 336 --frames 8                      分辨率与帧数
    --vis output_debug/DL3DV                      保存点云与图像（datasets/sample_utils/vis_util.visualize_gt）
    --no-getitem                                  只跑 loader 自己的 _get_views，不走 BaseDataset.__getitem__

每个样本先直接调用 _get_views（出错时打印完整 traceback，不会被重试掩盖），再走一遍 __getitem__（训练时的完整流程）。
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _find_block(cfg, cls_name, name=None, section="train_dataset"):
    sec = cfg[section]
    if name is not None:
        if name not in sec:
            raise KeyError(f"{name} 不在 {section} 里")
        return name, sec[name]
    for k in sec:
        v = sec[k]
        if hasattr(v, "get") and str(v.get("_target_", "")).split(".")[-1] == cls_name:
            return k, v
    return None, None


def _summ(x):
    x = np.asarray(x)
    v = x[np.isfinite(x) & (x > 0)]
    return (f"[{v.min():.3g}, {np.median(v):.3g}, {v.max():.3g}]" if v.size else "[empty]")


def _print_views(views, tag):
    for k, v in enumerate(views):
        img = v["img"]
        size = tuple(img.shape) if hasattr(img, "shape") else img.size
        d = np.asarray(v["depthmap"])
        valid = v.get("valid_mask", d > 0)
        ev = v.get("extra_valid", None)
        T = np.asarray(v.get("camera_pose", np.eye(4)))
        msg = (f"  [{tag} {k}] {v.get('label', '')}#{v.get('instance', '')} img={size} depth(min/med/max)={_summ(d)} "
               f"valid={np.mean(valid):.1%} det(R)={np.linalg.det(T[:3, :3]):.3f} "
               f"f={np.asarray(v['camera_intrinsics'])[0, 0]:.1f}")
        if ev is not None:
            msg += f" extra_valid={np.mean(ev):.1%}"
        print(msg, flush=True)


def debug_main(cls, argv=None, info=None):
    """info：传入一个 dict 时，把 len / ok / fail / 耗时 写进去（供 tools/check_datasets.py 汇总）。"""
    ap = argparse.ArgumentParser(description=f"debug {cls.__name__}")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/data/datamix_all_v9-5.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--section", default="train_dataset", choices=["train_dataset", "test_dataset"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--cache_file", default=None)
    ap.add_argument("--remap", action="append", default=[], metavar="SRC=DST")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336], metavar=("W", "H"))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-aug", action="store_true", help="保留配置里的 aug_crop / aug_focal（默认关掉，结果可复现）")
    ap.add_argument("--no-getitem", action="store_true")
    ap.add_argument("--vis", default=None, metavar="OUT_DIR")
    args = ap.parse_args(argv)

    import yaml
    from omegaconf import OmegaConf
    from datasets.base import fileio as fio
    from datasets.sample_utils.common import set_depth_filter

    kwargs = {}
    key = None
    if args.config and os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
        key, block = _find_block(cfg, cls.__name__, args.name, args.section)
        if block is not None:
            kwargs = OmegaConf.to_container(block, resolve=True)
    print(f"[debug] {cls.__name__}  config={args.config}  block={key or '(none, defaults only)'}", flush=True)
    kwargs.pop("_target_", None)
    kwargs.pop("transform", None)          # 配置里是 hydra partial；调试用默认 ToTensor
    if not args.keep_aug:
        kwargs["aug_crop"] = False
        kwargs["aug_focal"] = False
    for k in ("data_root", "cache_dir", "cache_file"):
        if getattr(args, k):
            kwargs[k] = getattr(args, k)
    if args.remap:
        rm = dict(kwargs.get("path_remap") or {})
        for item in args.remap:
            src, dst = item.split("=", 1)
            rm[src] = dst
        kwargs["path_remap"] = rm
    for item in args.set:
        k, v = item.split("=", 1)
        kwargs[k] = yaml.safe_load(v)
    if args.mode:
        kwargs["mode"] = args.mode
    kwargs.update(resolution=[list(args.res)], frame_num=args.frames, shuffle=False, seed=args.seed)
    print("[debug] kwargs: " + ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items())), flush=True)

    t0 = time.time()
    ds = cls(**kwargs)
    ds.frame_num = args.frames
    ds.convert_attributes()
    print(f"[debug] built in {time.time() - t0:.1f}s, len={len(ds)}, cache_dir={getattr(ds, 'cache_dir', None)}", flush=True)
    if info is not None:
        info.update(len=len(ds), build_s=round(time.time() - t0, 1))
    if len(ds) == 0:
        print("[debug] dataset is empty", flush=True)
        return 1

    idxs = args.idx if args.idx else sorted(set(np.linspace(0, len(ds) - 1, min(args.n, len(ds))).round().astype(int).tolist()))
    n_ok = n_fail = 0
    times = []
    for idx in idxs:
        print(f"[debug] ---- idx {idx}", flush=True)
        try:
            fio.set_path_remap(getattr(ds, "path_remap", None))
            set_depth_filter(getattr(ds, "pct_valid_only", False), getattr(ds, "pct_disable", False))
            t1 = time.time()
            views = ds._get_views(int(idx), list(args.res), np.random.default_rng(args.seed + int(idx)))
            times.append(time.time() - t1)
            _print_views(views, "raw")
            if not args.no_getitem:
                full = ds[(int(idx), 0, args.frames)]
                _print_views(full, "getitem")
                if args.vis:
                    from datasets.sample_utils.vis_util import visualize_gt
                    os.makedirs(args.vis, exist_ok=True)
                    visualize_gt(full, args.vis, prefix=f"{cls.__name__}_{idx}")
            n_ok += 1
        except Exception:
            n_fail += 1
            traceback.print_exc()
    if info is not None:
        info.update(ok=n_ok, fail=n_fail, get_views_s=round(float(np.mean(times)), 2) if times else None)
    print(f"[debug] done: ok={n_ok} fail={n_fail} mean _get_views time={np.mean(times) if times else float('nan'):.2f}s"
          + (f"  vis -> {args.vis}" if args.vis else ""), flush=True)
    return 0 if n_fail == 0 else 1
