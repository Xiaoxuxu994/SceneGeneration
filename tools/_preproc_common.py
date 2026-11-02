# -*- coding: utf-8 -*-
"""预处理脚本的公共部分：按训练用的同一份 datamix 配置构造数据集（同样的索引缓存、path_remap），
拿到全部序列 / 帧列表，再按场景多进程、场景内多线程地读写。"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_dataset(config, name):
    """按配置块实例化数据集（不做增强），返回 (dataset, 配置块 dict)。"""
    import importlib
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config)
    block = OmegaConf.to_container(cfg.train_dataset[name], resolve=True)
    target = block.pop("_target_")
    block.pop("transform", None)
    block.update(aug_crop=False, aug_focal=False, resolution=[[512, 336]], frame_num=8)
    mod, cls = target.rsplit(".", 1)
    ds = getattr(importlib.import_module(mod), cls)(**block)
    return ds, block


def run_scenes(fn, scenes, procs, desc):
    """fn(scene) -> (scene, status, n_frames, seconds)；多进程跑全部场景并打印进度。"""
    import multiprocessing as mp
    t0, done, n_frames, n_skip, n_err = time.time(), 0, 0, 0, 0
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        for scene, status, nf, dt in pool.imap_unordered(fn, scenes):
            done += 1
            n_frames += nf
            n_skip += status == "skip"
            n_err += status.startswith("error")
            if status.startswith("error") or done % 10 == 0 or done == len(scenes):
                el = time.time() - t0
                print(f"[{desc}] {done}/{len(scenes)} scenes ({n_skip} skipped, {n_err} errors), {n_frames} frames, "
                      f"{el / 60:.1f} min, eta {el / max(done - n_skip, 1) * (len(scenes) - done) / 60:.1f} min"
                      + (f" | {scene}: {status}" if status.startswith("error") else ""), flush=True)
    return n_err
