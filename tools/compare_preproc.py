#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比预处理数据与原始读法：同一个数据集、同一个 idx、同一个 seed（=> 同一组帧），分别用原始路径和 preproc_root 取样本，
逐帧比较相机、深度、图像，并记录两种读法的耗时。用来确认 tools/preprocess_megadepth.py / preprocess_gta_sfm.py
的输出可以直接替换原始数据进训练。

    python tools/compare_preproc.py --name MegaDepth --preproc obs://bucket-471/z00533370/dataset/MegaDepth_pre1024 --n 3
    python tools/compare_preproc.py --name GTA-sfm   --preproc obs://bucket-471/z00533370/dataset/GTA-SfM_frames --n 3

只用已经写完（有 _DONE）的场景。判定阈值（每帧都要满足）：
    位姿最大绝对差 < 1e-4；K 相对差 < 1%；
    深度：两边都有效的像素上相对误差中位数 < 1%，有效像素 IoU > 0.95；图像 PSNR > 25 dB。
MegaDepth 预处理先把图缩到长边 1024、深度最近邻缩放、JPEG 重新编码，所以图像和深度边缘允许有小差异；
GTA-SfM 是原样拆帧，应当几乎完全一致。
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import gt_audit as G  # noqa: E402  (会把工程根加入 sys.path)
from datasets.base import fileio as fio  # noqa: E402


def preproc_dir_of(ds, name, scene):
    if name == "MegaDepth":
        return osp.basename(str(ds.scene_to_path[scene]).rstrip("/"))
    return scene


def to_np_img(img):
    img = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    return img.transpose(1, 2, 0) if img.ndim == 3 and img.shape[0] == 3 else img


def compare_view(a, b):
    r = {}
    r["pose"] = float(np.abs(np.asarray(a["camera_pose"], np.float64) - np.asarray(b["camera_pose"], np.float64)).max())
    Ka, Kb = np.asarray(a["camera_intrinsics"], np.float64), np.asarray(b["camera_intrinsics"], np.float64)
    r["K_rel"] = float(np.abs(Ka - Kb).max() / max(np.abs(Ka).max(), 1e-9))
    da, db = np.asarray(a["depthmap"], np.float64), np.asarray(b["depthmap"], np.float64)
    if da.shape != db.shape:
        r["error"] = f"depth shape {da.shape} vs {db.shape}"
        return r
    va, vb = (da > 0) & np.isfinite(da), (db > 0) & np.isfinite(db)
    both = va & vb
    r["valid_iou"] = float(both.sum() / max((va | vb).sum(), 1))
    r["depth_rel_med"] = float(np.median(np.abs(da[both] - db[both]) / da[both])) if both.any() else float("nan")
    ia, ib = to_np_img(a["img"]).astype(np.float64), to_np_img(b["img"]).astype(np.float64)
    if ia.max() > 2:
        ia, ib = ia / 255.0, ib / 255.0
    mse = float(np.mean((ia - ib) ** 2))
    r["psnr"] = 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))
    r["ok"] = (r["pose"] < 1e-4 and r["K_rel"] < 0.01 and r["valid_iou"] > 0.95
               and (r["depth_rel_med"] < 0.01) and r["psnr"] > 25)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/datamix_all_v9-5-sub00.yaml")
    ap.add_argument("--name", required=True, help="配置里的数据集键名：MegaDepth / GTA-sfm")
    ap.add_argument("--preproc", required=True, help="预处理输出目录（本地或 obs://）")
    ap.add_argument("--n", type=int, default=3, help="对比几个样本（每个样本取自不同的已完成场景）")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg_all = OmegaConf.load(args.config)
    ds, _ = G.build_dataset(cfg_all, args.name, tuple(args.res), args.frames, "train", args.seed)
    print(f"[{args.name}] {len(ds.sequences)} sequences; looking for preprocessed scenes in {args.preproc}", flush=True)

    idxs = []
    for idx, scene in enumerate(ds.sequences):
        if fio.exists(osp.join(args.preproc, preproc_dir_of(ds, args.name, scene), "_DONE")):
            idxs.append(idx)
            if len(idxs) >= args.n:
                break
    if not idxs:
        print(f"[{args.name}] no finished scene (_DONE) under {args.preproc}", flush=True)
        return 1

    n_bad = 0
    for idx in idxs:
        seed = args.seed * 100003 + idx
        ds.preproc_root = None
        t = time.time(); orig = G.fetch_views(ds, idx, args.res, seed); t_orig = time.time() - t
        ds.preproc_root = args.preproc
        t = time.time(); pre = G.fetch_views(ds, idx, args.res, seed); t_pre = time.time() - t
        scene = ds.sequences[idx]
        print(f"\n== idx={idx} scene={scene}  original {t_orig:.1f}s  preproc {t_pre:.1f}s", flush=True)
        if len(orig) != len(pre):
            print(f"   FAIL: {len(orig)} vs {len(pre)} views", flush=True)
            n_bad += 1
            continue
        for k, (a, b) in enumerate(zip(orig, pre)):
            same = str(a.get("instance", k)) == str(b.get("instance", k))
            r = compare_view(a, b)
            if "error" in r:
                print(f"   view {k}: FAIL {r['error']}", flush=True)
                n_bad += 1
                continue
            ok = r["ok"] and same
            n_bad += not ok
            print(f"   view {k}: {'ok  ' if ok else 'FAIL'} frame={'same' if same else 'DIFF'} pose={r['pose']:.1e} "
                  f"K_rel={r['K_rel']:.1e} depth_rel_med={r['depth_rel_med']:.2e} valid_iou={r['valid_iou']:.3f} "
                  f"psnr={r['psnr']:.1f}dB", flush=True)
    print(f"\n[{args.name}] {'PASS' if n_bad == 0 else f'FAIL ({n_bad} views)'}", flush=True)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
