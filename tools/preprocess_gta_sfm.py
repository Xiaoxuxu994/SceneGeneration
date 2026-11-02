#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTA-SfM 预处理：把每个场景的 hdf5（平均 158MB）拆成按帧的小文件。

为什么：原始 loader 每个样本（8 帧）都要从 OBS 下载整个场景 hdf5 才能读出其中 8 帧，
训练时并发下单样本等待偶尔 20-30s，是 MegaDepth 之后第二慢的数据集；而且 h5 句柄只能串行读。
图像不重新编码（直接保存 h5 里的编码字节），深度默认保持 float32（GTA 远景深度大）。

输出（--out，本地或 obs://）：
    <out>/<scene>/image_<fid>.img   h5 里 image_<fid> 的原始编码字节（PIL 自动识别格式）
    <out>/<scene>/depth_<fid>.npy
    <out>/<scene>/cams.pkl          {fid(str): (K 3x3, pose 4x4)}
    <out>/<scene>/_DONE             场景写完的标记；重跑时跳过（可断点续跑）
训练时在 GTA-sfm 配置块加 preproc_root: <out> 即可，索引缓存不变。

用法（仓库根目录，和训练同一个环境）：
    python tools/preprocess_gta_sfm.py --out obs://bucket-471/z00533370/dataset/GTA-SfM_frames --procs 16
"""
import argparse
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _preproc_common import build_dataset, run_scenes  # noqa: E402

ARGS = None
DS = None


def process_scene(scene):
    from datasets.base import fileio as fio
    t0 = time.time()
    try:
        h5_path = DS.scene_to_path[scene]
        out_dir = osp.join(ARGS.out, scene)
        if not ARGS.overwrite and fio.exists(osp.join(out_dir, "_DONE")):
            return scene, "skip", 0, 0.0
        fids = [str(x) for x in DS.scene_meta[scene]["frame_ids"]]
        cams, jobs = {}, []
        with fio.h5_open(h5_path, "r") as f:            # 整个 h5 只下载一次
            for fid in fids:
                img = np.asarray(f[f"image_{fid}"]).reshape(-1).astype(np.uint8).tobytes()
                depth = np.asarray(f[f"depth_{fid}"]).astype(ARGS.depth_dtype)
                K = np.asarray(f[f"K_{fid}"]).astype(np.float32)
                pose = np.asarray(f[f"pose_{fid}"]).astype(np.float32)
                cams[fid] = (K, pose)
                jobs.append((osp.join(out_dir, f"image_{fid}.img"), img))
                jobs.append((osp.join(out_dir, f"depth_{fid}.npy"), fio.np_save_bytes(depth)))
        with ThreadPoolExecutor(ARGS.threads) as ex:
            list(ex.map(lambda pd: fio.write_bytes(*pd), jobs))
        fio.save_pickle(cams, osp.join(out_dir, "cams.pkl"))
        fio.write_bytes(osp.join(out_dir, "_DONE"), f"{len(cams)} frames\n".encode())
        return scene, "ok", len(cams), time.time() - t0
    except Exception as e:
        return scene, f"error {type(e).__name__}: {str(e)[:200]}", 0, time.time() - t0


def main():
    global ARGS, DS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/datamix_all_v9-5-sub00.yaml")
    ap.add_argument("--name", default="GTA-sfm", help="配置里的数据集键名")
    ap.add_argument("--out", required=True, help="输出目录（本地或 obs://）")
    ap.add_argument("--depth-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--procs", type=int, default=16, help="并行处理的场景数（每个进程同时持有一个解压后的 h5）")
    ap.add_argument("--threads", type=int, default=16, help="每个场景内并发写文件数")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个场景（试跑用）")
    ap.add_argument("--overwrite", action="store_true")
    ARGS = ap.parse_args()

    DS, block = build_dataset(ARGS.config, ARGS.name)
    scenes = list(DS.all_sequences)
    if ARGS.scenes:
        scenes = [s for s in scenes if s in set(ARGS.scenes)]
    if ARGS.limit > 0:
        scenes = scenes[:ARGS.limit]
    print(f"[GTA-sfm] data_root={block['data_root']}  {len(scenes)} scenes  -> {ARGS.out}  "
          f"depth={ARGS.depth_dtype} procs={ARGS.procs} threads={ARGS.threads}", flush=True)
    n_err = run_scenes(process_scene, scenes, ARGS.procs, "GTA-sfm")
    print(f"[GTA-sfm] finished, {n_err} scenes with errors（重跑同一条命令会只补做没完成的场景）", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
