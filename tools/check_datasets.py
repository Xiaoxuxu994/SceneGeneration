#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量读取检查：按 datamix 配置逐个构造数据集、读取若干样本，确认路径、缓存、文件解码和训练取样流程都正常。
每个数据集调用 datasets/base/debug.py 的 debug_main（与单独运行 python -m datasets.<x>_dataset 完全相同），
最后输出一张汇总表 <out>/check_summary.tsv。

用法（仓库根目录）：
    python tools/check_datasets.py --config configs/data/datamix_all_v9-5_obs.yaml --n 2
    python tools/check_datasets.py --config configs/data/datamix_all_v9-5_obs.yaml --datasets Hypersim CO3DV2 --vis out/vis
"""
import argparse
import importlib
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/datamix_all_v9-5_obs.yaml")
    ap.add_argument("--datasets", nargs="*", default=None, help="默认 weights 里的全部数据集")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336], metavar=("W", "H"))
    ap.add_argument("--vis", default=None, help="同时用 visualize_gt 保存点云与图像到 <vis>/<数据集>")
    ap.add_argument("--out", default="dataset_check")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from datasets.base.debug import debug_main
    cfg = OmegaConf.load(args.config)
    names = args.datasets or list(cfg.train_dataset.weights.keys())
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name in names:
        print(f"\n==================== {name} ====================", flush=True)
        info, status, err = {}, "OK", ""
        t0 = time.time()
        try:
            target = cfg.train_dataset[name]._target_
            mod, cls_name = target.rsplit(".", 1)
            cls = getattr(importlib.import_module(mod), cls_name)
            argv = ["--config", args.config, "--name", name, "--n", str(args.n), "--frames", str(args.frames),
                    "--res", str(args.res[0]), str(args.res[1])]
            if args.vis:
                argv += ["--vis", os.path.join(args.vis, name)]
            rc = debug_main(cls, argv, info=info)
            if rc != 0:
                status = "SAMPLE_FAIL" if info.get("len") else "EMPTY"
        except Exception as e:
            status, err = "BUILD_ERROR", f"{type(e).__name__}: {str(e)[:200]}"
            traceback.print_exc()
        rows.append((name, status, info.get("len", ""), info.get("ok", ""), info.get("fail", ""),
                     info.get("build_s", ""), info.get("get_views_s", ""), round(time.time() - t0, 1), err))
    head = ("dataset", "status", "len", "ok", "fail", "build_s", "get_views_s", "total_s", "error")
    lines = ["\t".join(head)] + ["\t".join(str(x) for x in r) for r in rows]
    with open(os.path.join(args.out, "check_summary.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    return 0 if all(r[1] == "OK" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
