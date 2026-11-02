#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MegaDepth 预处理：把训练要用的帧缩到长边 --max-side，写成按帧的小文件，并把每个场景的 COLMAP 相机预先解析好。

为什么：原始 loader 每个样本（8 帧）要从 OBS 读 8 张原始分辨率 jpg + 8 个原始分辨率 float32 h5 深度，
外加整份 sparse-txt/images.txt（含全部 2D 特征点，几十 MB），profile 实测每样本 172MB / 25s，
训练时并发下单样本等待 100-230s，是最慢的数据集。训练分辨率宽 400-600，长边 1024 足够。

输出（--out，本地或 obs://）：
    <out>/<scene>/images/<name>.jpg   RGB，长边 ≤ max_side（LANCZOS，quality 95）
    <out>/<scene>/depths/<name>.npy   深度，与图像同尺寸（最近邻缩放，不在边缘造飞点），默认 float16
    <out>/<scene>/cams.pkl            {name: (K 3x3，已按缩放调整；extrinsic 3x4，与 parse_colmap_to_dict 相同)}
    <out>/<scene>/_DONE               场景写完的标记；重跑时跳过已完成的场景（可断点续跑）
训练时在 MegaDepth 配置块加 preproc_root: <out> 即可，索引缓存不变。

用法（仓库根目录，和训练同一个环境）：
    python tools/preprocess_megadepth.py --out obs://bucket-471/z00533370/dataset/MegaDepth_pre1024 --procs 16
    python tools/preprocess_megadepth.py --out ... --scenes 0001 0015 --procs 2     # 先试几个场景
"""
import argparse
import os
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _preproc_common import build_dataset, run_scenes  # noqa: E402  （同时把仓库根目录加进 sys.path）

ARGS = None
DS = None


def _scale_K(K, sx, sy):
    K = np.array(K, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5      # 像素中心约定与 pi3.utils.cropping 一致
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def _one_frame(scene_path, out_dir, name, cams_raw):
    import io
    import cv2
    from PIL import Image
    from datasets.base import fileio as fio
    from datasets.base.utils import read_any_file

    img = fio.pil_open(osp.join(scene_path, "images", f"{name}.jpg")).convert("RGB")
    depth = np.asarray(read_any_file(osp.join(scene_path, "depths", f"{name}.h5")), dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    W, H = img.size
    s = min(1.0, ARGS.max_side / max(W, H))
    W2, H2 = (max(1, int(round(W * s))), max(1, int(round(H * s)))) if s < 1.0 else (W, H)
    if (W2, H2) != (W, H):
        img = img.resize((W2, H2), Image.LANCZOS)
    if depth.shape != (H2, W2):
        depth = cv2.resize(depth, (W2, H2), interpolation=cv2.INTER_NEAREST)
    K, extr = cams_raw[name]
    K2 = _scale_K(K, W2 / W, H2 / H)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=ARGS.jpeg_quality)
    fio.write_bytes(osp.join(out_dir, "images", f"{name}.jpg"), buf.getvalue())
    fio.write_bytes(osp.join(out_dir, "depths", f"{name}.npy"), fio.np_save_bytes(depth.astype(ARGS.depth_dtype)))
    return name, (K2.astype(np.float32), np.asarray(extr, dtype=np.float32))


def process_scene(scene):
    from datasets.base import fileio as fio
    from datasets.base.utils import parse_colmap_to_dict
    t0 = time.time()
    try:
        scene_path = DS.scene_to_path[scene]
        out_dir = osp.join(ARGS.out, osp.basename(str(scene_path).rstrip("/")))
        if not ARGS.overwrite and fio.exists(osp.join(out_dir, "_DONE")):
            return scene, "skip", 0, 0.0
        names = list(DS.scene_meta[scene]["frames_name"])
        cams_raw = parse_colmap_to_dict(osp.join(scene_path, "sparse-txt", "images.txt"),
                                        osp.join(scene_path, "sparse-txt", "cameras.txt"))
        missing = [n for n in names if n not in cams_raw]
        names = [n for n in names if n in cams_raw]
        with ThreadPoolExecutor(ARGS.threads) as ex:
            cams = dict(ex.map(lambda n: _one_frame(scene_path, out_dir, n, cams_raw), names))
        fio.save_pickle(cams, osp.join(out_dir, "cams.pkl"))
        fio.write_bytes(osp.join(out_dir, "_DONE"), f"{len(cams)} frames, {len(missing)} without camera\n".encode())
        return scene, "ok", len(cams), time.time() - t0
    except Exception as e:
        return scene, f"error {type(e).__name__}: {str(e)[:200]}", 0, time.time() - t0


def main():
    global ARGS, DS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/datamix_all_v9-5-sub00.yaml")
    ap.add_argument("--name", default="MegaDepth", help="配置里的数据集键名")
    ap.add_argument("--out", required=True, help="输出目录（本地或 obs://）")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--depth-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--procs", type=int, default=16, help="并行处理的场景数")
    ap.add_argument("--threads", type=int, default=8, help="每个场景内并发读写的帧数")
    ap.add_argument("--scenes", nargs="*", default=None, help="只处理这些场景（默认全部）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个场景（试跑用）")
    ap.add_argument("--overwrite", action="store_true", help="已完成（有 _DONE）的场景也重做")
    ARGS = ap.parse_args()

    DS, block = build_dataset(ARGS.config, ARGS.name)
    scenes = list(DS.all_sequences)
    if ARGS.scenes:
        scenes = [s for s in scenes if s in set(ARGS.scenes)]
    if ARGS.limit > 0:
        scenes = scenes[:ARGS.limit]
    print(f"[MegaDepth] data_root={block['data_root']}  {len(scenes)} scenes  -> {ARGS.out}  "
          f"max_side={ARGS.max_side} depth={ARGS.depth_dtype} procs={ARGS.procs} threads={ARGS.threads}", flush=True)
    n_err = run_scenes(process_scene, scenes, ARGS.procs, "MegaDepth")
    print(f"[MegaDepth] finished, {n_err} scenes with errors（重跑同一条命令会只补做没完成的场景）", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
