#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探查本地 / efs 上的 RealEstate10K 数据目录（如 /efs-mi-pay-gy1/n84398917/RE10K_HF/train），判断格式、数据量和是否可用：
  1. 全量扫描（多线程按子目录并行）：文件总数、总大小、后缀统计、目录深度、最大的几个文件；
  2. 按格式抽样解析，常见几种：
     - pixelSplat 的 .torch chunk：list[dict(url, timestamps, cameras[N,18], images[list of jpeg 字节], key)]，
       cameras 每行 = fx fy cx cy（归一化到 [0,1]）+ 2 个 0 + 3x4 w2c；
     - parquet / arrow（HF datasets）：打印 schema 和第一行各列的类型 / 长度；
     - 原始 txt（官方 RE10K 相机文件：首行 url，之后每行 timestamp + 18 个数）+ 图像目录；
     - 图像 / npy / json：打印形状、范围或 key；
  3. 对能拿到相机的场景做合法性检查：内参范围、R 是否正交（det≈1）、相邻帧平移量、帧数分布，并解码几张图看分辨率。

    python tools/probe_re10k.py /efs-mi-pay-gy1/n84398917/RE10K_HF/train --n 3
    python tools/probe_re10k.py <dir> --no-scan           # 只抽样解析，不全量扫描
"""
import argparse
import io
import json
import os
import os.path as osp
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def scan_dir(d):
    """递归扫描一个目录，返回 (文件数, 总字节, 后缀计数, 后缀字节, 最大深度, 最大的 5 个文件, 叶子目录文件数列表)。"""
    n, size, ext, ext_b, depth, big, leaf = 0, 0, Counter(), Counter(), 0, [], []
    stack = [(d, 0)]
    while stack:
        p, dep = stack.pop()
        k = 0
        try:
            it = os.scandir(p)
        except OSError:
            continue
        with it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append((e.path, dep + 1))
                else:
                    s = e.stat(follow_symlinks=False).st_size
                    x = osp.splitext(e.name)[1].lower() or "<none>"
                    n, size, k = n + 1, size + s, k + 1
                    ext[x] += 1
                    ext_b[x] += s
                    depth = max(depth, dep)
                    big = sorted(big + [(s, e.path)], reverse=True)[:5]
        if k:
            leaf.append(k)
    return n, size, ext, ext_b, depth, big, leaf


def full_scan(root, workers):
    t0 = time.time()
    top = sorted(os.scandir(root), key=lambda e: e.name)
    dirs = [e.path for e in top if e.is_dir()]
    files = [e for e in top if not e.is_dir()]
    print(f"[root] {root}: {len(dirs)} dirs, {len(files)} files at top level")
    names = [e.name for e in top]
    print(f"  first: {names[:5]}\n  last:  {names[-5:]}")
    n, size, ext, ext_b, depth, big, leaf = len(files), 0, Counter(), Counter(), 0, [], [len(files)] if files else []
    for e in files:
        s = e.stat().st_size
        x = osp.splitext(e.name)[1].lower() or "<none>"
        size += s
        ext[x] += 1
        ext_b[x] += s
        big = sorted(big + [(s, e.path)], reverse=True)[:5]
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(scan_dir, dirs):
            n += r[0]; size += r[1]; ext.update(r[2]); ext_b.update(r[3])
            depth = max(depth, r[4] + 1); big = sorted(big + r[5], reverse=True)[:5]; leaf += r[6]
            done += 1
            if done % 500 == 0:
                print(f"  ... scanned {done}/{len(dirs)} dirs, {n} files, {human(size)}, {time.time() - t0:.0f}s", flush=True)
    print(f"[scan] {n} files, {human(size)}, max depth {depth}, {time.time() - t0:.0f}s")
    for x, c in ext.most_common(10):
        print(f"  {x:8s} {c:>10d} files  {human(ext_b[x]):>10s}  avg {human(ext_b[x] / c)}")
    print("  largest: " + ", ".join(f"{osp.relpath(p, root)} ({human(s)})" for s, p in big))
    if leaf:
        a = np.array(leaf)
        print(f"  files per leaf dir: {len(a)} dirs, min {a.min()} median {int(np.median(a))} max {a.max()}")
    return ext


def find_samples(root, exts, n):
    """每种后缀找前 n 个文件（按目录顺序 DFS，找够就停）。"""
    want = {x: [] for x in exts}
    stack = [root]
    while stack and any(len(v) < n for v in want.values()):
        p = stack.pop()
        try:
            es = sorted(os.scandir(p), key=lambda e: e.name, reverse=True)
        except OSError:
            continue
        for e in es:
            if e.is_dir():
                stack.append(e.path)
            else:
                x = osp.splitext(e.name)[1].lower() or "<none>"
                if x in want and len(want[x]) < n:
                    want[x].append(e.path)
    return want


def decode_img(b):
    from PIL import Image
    im = Image.open(io.BytesIO(bytes(b)))
    return im.size, im.mode


def check_cams(cams, tag):
    """cams: [N,18] pixelSplat / 官方 RE10K 格式（fx fy cx cy 0 0 + 3x4 w2c）。"""
    cams = np.asarray(cams, dtype=np.float64)
    if cams.ndim != 2 or cams.shape[1] != 18:
        print(f"    [{tag}] cameras shape {cams.shape}, not [N,18]; first row {cams.reshape(-1)[:20]}")
        return
    K, E = cams[:, :4], cams[:, 6:].reshape(-1, 3, 4)
    R, t = E[:, :, :3], E[:, :, 3]
    det = np.linalg.det(R)
    orth = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    c = -np.einsum("nji,nj->ni", R, t)               # 相机中心 = -R^T t
    step = np.linalg.norm(np.diff(c, axis=0), axis=1) if len(c) > 1 else np.zeros(1)
    print(f"    [{tag}] {len(cams)} frames | fx {K[:, 0].min():.3f}-{K[:, 0].max():.3f} fy {K[:, 1].min():.3f}-{K[:, 1].max():.3f} "
          f"cx {K[:, 2].mean():.3f} cy {K[:, 3].mean():.3f} (归一化内参应在 0~2 左右, cx/cy≈0.5) | cols4-5 {np.abs(cams[:, 4:6]).max():.2g}")
    print(f"    [{tag}] det(R) {det.min():.4f}~{det.max():.4f}, max|RR^T-I| {orth:.2e} | 相机中心跨度 {np.ptp(c, axis=0).round(3).tolist()}, "
          f"相邻帧位移 median {np.median(step):.4f} max {step.max():.4f} | nan {int(np.isnan(cams).sum())}")


def peek_torch(path, n):
    import torch
    t0 = time.time()
    d = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  [torch] {path} ({human(osp.getsize(path))}, load {time.time() - t0:.1f}s): {type(d).__name__}"
          + (f" len {len(d)}" if hasattr(d, "__len__") else ""))
    items = d if isinstance(d, list) else [d]
    nf = [len(x["cameras"]) for x in items if isinstance(x, dict) and "cameras" in x]
    if nf:
        print(f"    scenes in chunk {len(items)}, frames/scene min {min(nf)} median {int(np.median(nf))} max {max(nf)} total {sum(nf)}")
    for x in items[:n]:
        if not isinstance(x, dict):
            print(f"    item {type(x).__name__}")
            continue
        desc = {k: (f"{type(v).__name__}{tuple(v.shape)} {v.dtype}" if hasattr(v, "shape") else
                    f"{type(v).__name__} len {len(v)}" if isinstance(v, (list, tuple)) else repr(v)[:80]) for k, v in x.items()}
        print(f"    keys: {desc}")
        if "cameras" in x:
            check_cams(x["cameras"], x.get("key", "?"))
        if "images" in x and len(x["images"]):
            ims = x["images"]
            idx = sorted({0, len(ims) // 2, len(ims) - 1})
            info = []
            for i in idx:
                b = ims[i].numpy().tobytes() if hasattr(ims[i], "numpy") else ims[i]
                info.append(f"#{i} {decode_img(b)} {human(len(b))}")
            print(f"    images: {len(ims)} encoded, decoded " + "; ".join(info))
        if "timestamps" in x:
            ts = np.asarray(x["timestamps"], dtype=np.int64)
            dt = np.diff(ts) if len(ts) > 1 else np.zeros(1)
            print(f"    timestamps: {ts[:3].tolist()}..., step median {np.median(dt):.0f} (us), monotonic {bool((dt > 0).all())}")
    return sum(nf), len(items)


def peek_parquet(path):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    print(f"  [parquet] {path} ({human(osp.getsize(path))}): {f.metadata.num_rows} rows, {f.metadata.num_row_groups} row groups")
    print(f"    schema: {f.schema_arrow}")
    row = f.read_row_group(0).slice(0, 1).to_pylist()[0]
    for k, v in row.items():
        if isinstance(v, (bytes, bytearray)):
            try:
                print(f"    {k}: bytes {human(len(v))}, image {decode_img(v)}")
            except Exception:
                print(f"    {k}: bytes {human(len(v))}")
        elif isinstance(v, list):
            print(f"    {k}: list len {len(v)}, first {repr(v[0])[:120] if v else None}")
            if v and isinstance(v[0], (bytes, bytearray, dict)):
                b = v[0].get("bytes") if isinstance(v[0], dict) else v[0]
                if b:
                    try:
                        print(f"      first element image {decode_img(b)}")
                    except Exception:
                        pass
            if k in ("cameras", "camera", "poses") and v and isinstance(v[0], list):
                check_cams(v, k)
        elif isinstance(v, dict):
            print(f"    {k}: dict keys {list(v)[:10]}")
            if "bytes" in v and v["bytes"]:
                print(f"      image {decode_img(v['bytes'])}")
        else:
            print(f"    {k}: {repr(v)[:150]}")
    return f.metadata.num_rows


def peek_txt(path):
    lines = open(path, errors="replace").read().splitlines()
    print(f"  [txt] {path}: {len(lines)} lines; head: {lines[:3]}")
    try:
        rows = np.array([[float(v) for v in l.split()] for l in lines[1:] if l.strip()])
        if rows.ndim == 2 and rows.shape[1] == 19:      # 官方格式：timestamp + 18
            check_cams(rows[:, 1:], osp.basename(path))
    except ValueError:
        pass


def peek_other(path, x):
    if x in (".jpg", ".jpeg", ".png", ".webp"):
        print(f"  [img] {path}: {decode_img(open(path, 'rb').read())}")
    elif x == ".npy":
        a = np.load(path, allow_pickle=True)
        print(f"  [npy] {path}: {a.shape} {a.dtype}" + (f" range {a.min():.3g}~{a.max():.3g}" if a.dtype.kind in "fiu" and a.size else ""))
    elif x == ".json" and osp.getsize(path) < 50 << 20:
        d = json.load(open(path))
        print(f"  [json] {path}: {type(d).__name__} " + (f"keys {list(d)[:15]}" if isinstance(d, dict) else f"len {len(d)}, first {repr(d[0])[:200] if d else None}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="/efs-mi-pay-gy1/n84398917/RE10K_HF/train")
    ap.add_argument("--n", type=int, default=2, help="每种后缀抽样解析几个文件")
    ap.add_argument("--workers", type=int, default=32, help="全量扫描的线程数")
    ap.add_argument("--no-scan", action="store_true", help="跳过全量扫描")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    root = args.root.rstrip("/")
    print(f"[probe] {root}")
    parent = osp.dirname(root)
    print(f"[parent] {parent}: {sorted(os.listdir(parent))[:30]}")

    if args.no_scan:
        ext = Counter()
        for _, _, fs in os.walk(root):
            ext.update(osp.splitext(f)[1].lower() or "<none>" for f in fs)
            if sum(ext.values()) > 2000:
                break
        print(f"[ext sample] {dict(ext.most_common(10))}")
    else:
        ext = full_scan(root, args.workers)

    exts = [x for x, _ in ext.most_common(8)]
    samples = find_samples(root, exts, args.n)
    print("\n[samples]")
    for x, paths in samples.items():
        for p in paths:
            try:
                if x in (".torch", ".pt", ".pth"):
                    peek_torch(p, args.n)
                elif x == ".parquet":
                    peek_parquet(p)
                elif x == ".txt":
                    peek_txt(p)
                else:
                    peek_other(p, x)
            except Exception as e:
                print(f"  [fail] {p}: {type(e).__name__}: {str(e)[:200]}")
    # 有 index.json（pixelSplat 的 key -> chunk 文件名）时统计场景数
    for idx in (osp.join(root, "index.json"), osp.join(parent, "index.json")):
        if osp.exists(idx):
            d = json.load(open(idx))
            print(f"\n[index] {idx}: {len(d)} entries, {len(set(d.values())) if isinstance(d, dict) else '-'} chunks")
    print("[probe] done")


if __name__ == "__main__":
    main()
