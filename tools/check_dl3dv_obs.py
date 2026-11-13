#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL3DV（obs://bucket-471/y00959808/data/sr_datasets/processed/dl3dv，布局 <1K..7K>/<hash>/dense/rgb/frame_xxxxx.png）
的场景数、RGB、相机检查，不看深度。相机来自 pkl_map/meta_data.pkl 的 data_store：
    {hash: [{"img_path": "<批次>/<hash>/dense/rgb/frame_00001.png", "extri": "<4x4 文本>", "intri": "<3x3 文本>", ...}, ...]}
（manifest_*.jsonl 只覆盖 630 个场景且路径是 images_8/，与实际文件不符，不用；extri 与 manifest 的 c2w 数值相同。）

  1) 全量：逐个场景列 dense/ 和 dense/rgb/（多线程，不读帧），与 meta_data.pkl 的帧逐个对名字 -> <out>/dl3dv_scenes.csv
  2) 抽查（含 --scenes 指定的）：读第一帧 png 的尺寸，对照内参主点；检查 extri 是合法刚体变换；打印相机中心轨迹范围

    python tools/check_dl3dv_obs.py --scenes 5K/a85c949d27a67ebded3ed23a20a911fd6a751c1eae945dc4d10d0afe442e1294 --n 3
    python tools/check_dl3dv_obs.py --scan-only      # 只扫目录：场景数、每个场景的 dense/rgb 帧数（约 1 分钟），不读 meta_data.pkl
"""
import argparse
import csv
import os
import pickle
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "obs://bucket-471/y00959808/data/sr_datasets/processed/dl3dv"


def mox():
    import moxing
    return moxing


def ls(p):
    return sorted(x.rstrip("/").split("/")[-1] for x in mox().file.list_directory(p))


def mat(s):
    """meta_data.pkl 里矩阵是 numpy 的文本形式 "[[a b c]\n [d e f]]"。"""
    if not isinstance(s, str):
        return np.asarray(s, dtype=np.float64)
    rows = [r for r in s.replace("[", " ").replace("]", "\n").split("\n") if r.strip()]
    return np.array([[float(x) for x in r.split()] for r in rows], dtype=np.float64)


def scan_scene(batch, h, cam_frames):
    d = f"{ROOT}/{batch}/{h}"
    try:
        dense = ls(f"{d}/dense")
        rgb = ls(f"{d}/dense/rgb") if "rgb" in dense else []
        cam_names = {os.path.basename(f["img_path"]) for f in cam_frames}
        rgb_set = set(rgb)
        return dict(batch=batch, hash=h, n_rgb=len(rgb), n_cam=len(cam_frames),
                    n_matched=len(cam_names & rgb_set), dense=";".join(dense), err="")
    except Exception as e:
        return dict(batch=batch, hash=h, n_rgb=0, n_cam=len(cam_frames), n_matched=0, dense="",
                    err=f"{type(e).__name__}: {str(e)[:100]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=[], help="一定要抽查的场景，写成 <批次>/<hash>")
    ap.add_argument("--n", type=int, default=3, help="另外随机抽查几个场景")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default="/efs-mi-pay-gy1/z00533370/exp/logs/dl3dv_check")
    ap.add_argument("--scan-only", action="store_true", help="不读 meta_data.pkl（加载要 7 分钟），只统计场景数和 RGB 帧数")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    if args.scan_only:
        batches = [b for b in ls(ROOT) if b.endswith("K") and b[:-1].isdigit()]
        jobs = [(b, h) for b in batches for h in ls(f"{ROOT}/{b}")]
        print(f"[scan] {len(jobs)} scenes: " + ", ".join(f"{b} {sum(j[0] == b for j in jobs)}" for b in batches)
              + f" ({time.time() - t0:.0f}s); counting dense/rgb frames ...", flush=True)
        with ThreadPoolExecutor(args.threads) as ex:
            rows = list(ex.map(lambda j: scan_scene(j[0], j[1], []), jobs))
        with open(f"{args.out}/dl3dv_scenes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["batch", "hash", "n_rgb", "dense", "err"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        n = np.array([r["n_rgb"] for r in rows])
        print(f"[scan] done in {time.time() - t0:.0f}s -> {args.out}/dl3dv_scenes.csv")
        print(f"  dense/ contents (how many scenes have each): {dict(Counter(e for r in rows for e in r['dense'].split(';') if e))}")
        print(f"  scenes with rgb frames: {(n > 0).sum()}/{len(rows)}; frames total {n.sum()}; per scene min/median/max "
              f"{n.min()}/{int(np.median(n))}/{n.max()}")
        for b in batches:
            nb = np.array([r["n_rgb"] for r in rows if r["batch"] == b])
            print(f"  {b}: {len(nb)} scenes, {nb.sum()} frames")
        errs = [r for r in rows if r["err"]]
        if errs:
            print(f"  list errors: {len(errs)}, e.g. {errs[:2]}")
        return 0
    meta = pickle.loads(mox().file.read(f"{ROOT}/pkl_map/meta_data.pkl", binary=True))
    store = meta["data_store"] if isinstance(meta, dict) and "data_store" in meta else meta
    other = {k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in meta.items() if k != "data_store"} \
        if isinstance(meta, dict) else {}
    first = next(iter(store.values()))[0]
    print(f"[meta_data.pkl] {len(store)} scenes, other keys {other}; frame keys {list(first)} ({time.time() - t0:.0f}s)", flush=True)

    batches = [b for b in ls(ROOT) if b.endswith("K") and b[:-1].isdigit()]
    jobs = [(b, h) for b in batches for h in ls(f"{ROOT}/{b}")]
    print(f"[scan] {len(jobs)} scene dirs in {batches}; listing dense/ and dense/rgb/ ...", flush=True)
    with ThreadPoolExecutor(args.threads) as ex:
        rows = list(ex.map(lambda j: scan_scene(j[0], j[1], store.get(j[1], [])), jobs))
    with open(f"{args.out}/dl3dv_scenes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_rgb = np.array([r["n_rgb"] for r in rows])
    n_cam = np.array([r["n_cam"] for r in rows])
    full = sum(r["n_rgb"] > 0 and r["n_matched"] == r["n_cam"] == r["n_rgb"] for r in rows)
    print(f"[scan] done in {time.time() - t0:.0f}s -> {args.out}/dl3dv_scenes.csv", flush=True)
    print(f"  dense/ contents (how many scenes have each): {dict(Counter(e for r in rows for e in r['dense'].split(';') if e))}")
    print(f"  scenes with rgb: {(n_rgb > 0).sum()}/{len(rows)}; rgb frames total {n_rgb.sum()}, "
          f"per scene min/median/max {n_rgb.min()}/{int(np.median(n_rgb))}/{n_rgb.max()}")
    print(f"  camera frames total {n_cam.sum()}; scenes where every camera frame has its png and vice versa: {full}/{len(rows)}")
    bad = [r for r in rows if not (r["n_rgb"] > 0 and r["n_matched"] == r["n_cam"] == r["n_rgb"])]
    for r in bad[:10]:
        print(f"  mismatch {r['batch']}/{r['hash'][:16]}… rgb {r['n_rgb']} cam {r['n_cam']} matched {r['n_matched']} {r['err']}")
    print(f"  scenes not in meta_data.pkl: {sum(r['n_cam'] == 0 for r in rows)}")

    import cv2
    pick = list(args.scenes)
    rest = [f"{r['batch']}/{r['hash']}" for r in rows if r["n_rgb"] > 0]
    random.Random(0).shuffle(rest)
    pick += [s for s in rest if s not in pick][: args.n]
    for s in pick:
        b, h = s.split("/")
        d = f"{ROOT}/{s}"
        print(f"\n[sample] {s}", flush=True)
        try:
            print(f"  dense/: {ls(f'{d}/dense')}")
            fr = store.get(h, [])
            if not fr:
                print("  not in meta_data.pkl")
                continue
            im = cv2.imdecode(np.frombuffer(mox().file.read(f"{ROOT}/{fr[0]['img_path']}", binary=True), np.uint8),
                              cv2.IMREAD_UNCHANGED)
            K = mat(fr[0]["intri"])
            H, W = im.shape[:2]
            print(f"  frames {len(fr)}, image {im.shape} {im.dtype}; K fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} "
                  f"(W/2={W / 2:.1f}, H/2={H / 2:.1f}); HFOV {np.degrees(2 * np.arctan(W / 2 / K[0, 0])):.1f} deg")
            Ks = np.stack([mat(f["intri"]) for f in fr])
            print(f"  K identical across frames: {bool(np.allclose(Ks, Ks[0]))}")
            E = np.stack([mat(f["extri"]) for f in fr])
            R = E[:, :3, :3]
            orth = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
            print(f"  extri {E.shape}: max |R R^T - I| = {orth:.1e}, det(R) {np.linalg.det(R).min():.4f}..{np.linalg.det(R).max():.4f}, "
                  f"last row ok {bool(np.allclose(E[:, 3], [0, 0, 0, 1]))}")
            c = E[:, :3, 3]                              # 当作 c2w 时的相机中心
            step = np.linalg.norm(np.diff(c, axis=0), axis=1)
            print(f"  camera centers (as c2w): range {np.ptp(c, axis=0).round(3)}, path length {step.sum():.3f}, "
                  f"median step {np.median(step):.4f}")
            print(f"  other keys of a frame: { {k: str(v)[:80] for k, v in fr[0].items() if k not in ('img_path', 'extri', 'intri')} }")
        except Exception as e:
            print(f"  {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    sys.exit(main())
