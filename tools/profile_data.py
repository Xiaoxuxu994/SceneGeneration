#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据读取耗时分析：训练时每张卡每步要 1 个样本（8 帧），所有卡互相等，最慢的那张卡的取数时间决定整步耗时。
这个脚本按训练时的设置（同一份 datamix 配置、数据增强、过滤开关、随机分辨率）构造数据集，测：

  1) 逐数据集（--n 个样本）：BaseDataset.__getitem__ 完整耗时（含视差不足重选帧、失败重试），
     其中 OBS 读取的次数 / 字节 / 耗时，_get_views 调用次数，重选帧与失败次数；
  2) 按配置权重混合后的期望：平均每个样本多少秒、每张卡 W 个 worker 时每步至少要等多少秒，
     以及 N 张卡时"某张卡这一步抽到慢数据集"的概率；
  3) 可选 --procs P --seconds S：P 个进程同时按权重混合取样（模拟一个节点上 8 卡 × worker 并发读 OBS），
     看并发下每秒能出多少样本、单样本耗时是否被拖慢。

用法（仓库根目录，和训练同一个环境；缓存先拷到本地盘更接近训练）：
    DATASET_CACHE_DIR=/cache/dataset_cache python tools/profile_data.py --n 6
    python tools/profile_data.py --datasets MegaDepth ASE --n 10
    python tools/profile_data.py --n 4 --procs 16 --seconds 300 --workers 4 --gpus 128
输出：<out>/profile_per_dataset.tsv，终端打印汇总表。
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# OBS 读取计时：fileio 里所有 obs:// 读取都经过 read_bytes
# ---------------------------------------------------------------------------
IO = defaultdict(float)


def _patch_io():
    from datasets.base import fileio as fio
    orig = fio.read_bytes

    def timed(path):
        t = time.time()
        data = orig(path)
        IO["t"] += time.time() - t
        IO["n"] += 1
        IO["bytes"] += len(data) if data is not None else 0
        return data

    fio.read_bytes = timed


def _training_resolutions(cfg_train, seed=0):
    from datasets.base.base_dataset import sample_resolutions
    return sample_resolutions(
        aspect_ratio_range=cfg_train.aspect_ratio_range, pixel_count_range=cfg_train.pixel_count_range,
        patch_size=cfg_train.patch_size, num_resolutions=-1, seed=seed,
        width_range=cfg_train.get("width_range", None))


def build(cfg, name, resolutions):
    import hydra
    ds = hydra.utils.instantiate(cfg.train_dataset[name], resolution=resolutions)
    ds.convert_attributes()
    return ds


def one_sample(ds, rng, frames, n_res):
    idx = int(rng.integers(0, len(ds)))
    ar = int(rng.integers(0, n_res))
    c0 = (ds._n_load, ds._n_low_parallax, ds._n_fail)
    io0 = dict(IO)
    t = time.time()
    err = None
    try:
        ds[(idx, ar, frames)]
    except Exception as e:  # 记录下来，不中断
        err = f"{type(e).__name__}: {str(e)[:120]}"
    dt = time.time() - t
    return dict(
        time=dt,
        get_views=ds._n_load - c0[0], low_parallax=ds._n_low_parallax - c0[1], fail=ds._n_fail - c0[2],
        io_t=IO["t"] - io0.get("t", 0.0), io_n=IO["n"] - io0.get("n", 0.0),
        io_mb=(IO["bytes"] - io0.get("bytes", 0.0)) / 1e6, err=err)


def _mix_worker(args):
    """并发测试的子进程：按权重抽数据集取样，跑 seconds 秒。"""
    (wid, names, probs, seconds, frames, n_res) = args
    rng = np.random.default_rng(1000 + wid)
    np.random.seed(1000 + wid)
    for ds in _DS.values():
        ds._rng = np.random.default_rng(2000 + wid)
    out, t_end = [], time.time() + seconds
    while time.time() < t_end:
        name = names[int(rng.choice(len(names), p=probs))]
        r = one_sample(_DS[name], rng, frames, n_res)
        out.append((name, r["time"]))
    return out


_DS = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/datamix_all_v9-5-sub00.yaml")
    ap.add_argument("--train-config", default="configs/train/train_pi3_highres.yaml",
                    help="取随机分辨率的设置（宽 400-600 等）")
    ap.add_argument("--datasets", nargs="*", default=None, help="默认 weights 里的全部数据集")
    ap.add_argument("--n", type=int, default=6, help="每个数据集测多少个样本")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4, help="训练时每张卡的 dataloader worker 数（用于估算）")
    ap.add_argument("--gpus", type=int, default=128, help="训练卡数（用于估算慢样本拖慢整步的概率）")
    ap.add_argument("--procs", type=int, default=0, help=">0 时做并发测试：多少个进程同时取样（模拟一个节点上 卡数 × worker）")
    ap.add_argument("--cards-per-node", type=int, default=8, help="并发测试折算到每张卡的供数速度")
    ap.add_argument("--step-s", type=float, default=2.4, help="模型每步计算耗时（8 卡冒烟实测约 2.4s），用来判断数据是否供得上")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="profile_data_out")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cfg = OmegaConf.load(args.config)
    tcfg = OmegaConf.load(args.train_config).train
    weights = dict(cfg.train_dataset.weights)
    names = args.datasets or list(weights.keys())
    resolutions = _training_resolutions(tcfg, args.seed)
    print(f"[profile] config={args.config}  {len(names)} datasets  {len(resolutions)} resolutions  "
          f"frames={args.frames}  n={args.n}", flush=True)
    _patch_io()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for name in names:
        t0 = time.time()
        try:
            ds = build(cfg, name, resolutions)
        except Exception as e:
            print(f"[{name}] BUILD_ERROR {type(e).__name__}: {e}", flush=True)
            continue
        build_s = time.time() - t0
        _DS[name] = ds
        rs = [one_sample(ds, rng, args.frames, len(resolutions)) for _ in range(args.n)]
        tm = np.array([r["time"] for r in rs])
        row = dict(
            dataset=name, weight=weights.get(name, 0), build_s=round(build_s, 1),
            mean_s=tm.mean(), p50_s=np.median(tm), max_s=tm.max(),
            io_share=sum(r["io_t"] for r in rs) / max(tm.sum(), 1e-9),
            io_n=np.mean([r["io_n"] for r in rs]), io_mb=np.mean([r["io_mb"] for r in rs]),
            get_views=np.mean([r["get_views"] for r in rs]),
            low_parallax=np.mean([r["low_parallax"] for r in rs]), fail=np.mean([r["fail"] for r in rs]),
            errors=sum(r["err"] is not None for r in rs))
        rows.append(row)
        print(f"[{name:16s}] build {build_s:6.1f}s | sample mean {row['mean_s']:6.2f}s p50 {row['p50_s']:6.2f}s "
              f"max {row['max_s']:6.2f}s | OBS {row['io_n']:5.0f} reads {row['io_mb']:6.1f}MB "
              f"({row['io_share']:4.0%} of time) | _get_views x{row['get_views']:.1f} "
              f"reselect {row['low_parallax']:.1f} fail {row['fail']:.1f}"
              + (f" | {row['errors']} errors" if row["errors"] else ""), flush=True)

    if not rows:
        return
    rows.sort(key=lambda r: -r["mean_s"])
    keys = ["dataset", "weight", "build_s", "mean_s", "p50_s", "max_s", "io_share", "io_n", "io_mb",
            "get_views", "low_parallax", "fail", "errors"]
    with open(os.path.join(args.out, "profile_per_dataset.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    # ---------------- 混合期望 ----------------
    w = np.array([r["weight"] for r in rows], float)
    if w.sum() > 0:
        p = w / w.sum()
        mean_mix = float((p * np.array([r["mean_s"] for r in rows])).sum())
        print("\n==> 按权重混合（训练时每张卡每步 1 个样本）")
        print(f"    平均每个样本 {mean_mix:.2f}s；每张卡 {args.workers} 个 worker 时，每步至少要等 "
              f"{mean_mix / args.workers:.2f}s 才能稳定供上数据（模型计算约 2.4s/步）")
        for thr in (10, 30, 60):
            q = float(p[np.array([r["max_s"] for r in rows]) >= thr].sum())   # 抽到"最慢样本 ≥ thr 秒"的数据集的概率
            hit = 1 - (1 - q) ** args.gpus
            print(f"    含 ≥{thr:>2d}s 慢样本的数据集合计占 {q:5.1%}；{args.gpus} 张卡同一步至少一张抽到它们的概率 {hit:5.1%}")
        print("    贡献最大的数据集（权重占比 × 平均耗时）：")
        contrib = sorted(((float(pi * r["mean_s"]), r["dataset"], float(pi), r["mean_s"]) for pi, r in zip(p, rows)), reverse=True)
        for c, n, pi, m in contrib[:6]:
            print(f"      {n:16s} 占比 {pi:5.1%} × {m:6.2f}s = {c:5.2f}s ({c / mean_mix:4.0%})")

    # ---------------- 并发测试 ----------------
    if args.procs > 0 and w.sum() > 0:
        import multiprocessing as mp
        ns = [r["dataset"] for r in rows]
        ctx = mp.get_context("fork")
        print(f"\n==> 并发测试：{args.procs} 个进程同时取样 {args.seconds}s（按权重混合）", flush=True)
        t0 = time.time()
        with ctx.Pool(args.procs) as pool:
            res = pool.map(_mix_worker, [(i, ns, p, args.seconds, args.frames, len(resolutions)) for i in range(args.procs)])
        wall = time.time() - t0
        allr = [x for r in res for x in r]
        per = defaultdict(list)
        for n, t in allr:
            per[n].append(t)
        tt = np.array([t for _, t in allr])
        print(f"    共 {len(allr)} 个样本，{len(allr) / wall:.2f} 样本/秒；单样本 mean {tt.mean():.2f}s "
              f"p50 {np.median(tt):.2f}s p95 {np.percentile(tt, 95):.2f}s max {tt.max():.2f}s")
        per_card = len(allr) / wall / max(args.cards_per_node, 1)
        need = 1.0 / args.step_s
        print(f"    折算到每张卡（{args.cards_per_node} 卡/节点）：{per_card:.3f} 样本/秒；模型每步 {args.step_s}s 需要 ≥{need:.3f} 样本/秒 "
              f"-> {'供得上' if per_card >= need else f'供不上，数据侧每步约 {1.0 / max(per_card, 1e-9):.1f}s'}")
        print("    与单进程相比（并发下变慢说明 OBS / CPU 成为瓶颈）：")
        single = {r["dataset"]: r["mean_s"] for r in rows}
        for n in sorted(per, key=lambda k: -np.mean(per[k]))[:8]:
            print(f"      {n:16s} 并发 mean {np.mean(per[n]):6.2f}s (n={len(per[n])})  单进程 {single[n]:6.2f}s")
    print(f"\n==> 结果：{os.path.join(args.out, 'profile_per_dataset.tsv')}")


if __name__ == "__main__":
    main()
