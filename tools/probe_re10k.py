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
    python tools/probe_re10k.py obs://bucket-471/dataset/re10k --chunks 20     # OBS 上的 pixelSplat chunk：抽 20 个 chunk 统计
    python tools/probe_re10k.py obs://bucket-471/dataset/re10k --chunks 0 --procs 32   # 下载全部 chunk 精确统计帧数
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


def mox():
    import moxing
    return moxing


def chunk_stats(job):
    """下载一个 OBS chunk 到本地、torch.load，返回每个视频切片的统计；出错返回错误信息。"""
    src, local_tmp = job
    import torch
    local = osp.join(local_tmp, src.replace("/", "_").replace(":", ""))
    t0 = time.time()
    try:
        mox().file.copy(src, local)
        size = osp.getsize(local)
        items = torch.load(local, map_location="cpu", weights_only=False)
        rows = []
        for x in items:
            cams = np.asarray(x["cameras"], dtype=np.float64)
            ims = x["images"]
            b = ims[0].numpy().tobytes() if len(ims) else b""
            wh = decode_img(b)[0] if b else None
            R = cams[:, 6:].reshape(-1, 3, 4)[:, :, :3] if cams.ndim == 2 and cams.shape[1] == 18 else None
            bad = (R is None or np.isnan(cams).any() or np.abs(np.linalg.det(R) - 1).max() > 1e-2
                   or len(ims) != len(cams) or len(x["timestamps"]) != len(cams))
            ts = np.asarray(x["timestamps"], dtype=np.int64)
            rows.append(dict(key=x["key"], n=len(ims), wh=wh, fx=float(cams[0, 0]) if cams.size else None, bad=bool(bad),
                             dur=float(ts[-1] - ts[0]) / 1e6 if len(ts) > 1 else 0.0, jpg_bytes=len(b)))
        return src, size, rows, None, time.time() - t0
    except Exception as e:
        return src, 0, [], f"{type(e).__name__}: {str(e)[:150]}", time.time() - t0
    finally:
        if osp.exists(local):
            os.remove(local)


def obs_probe(args):
    from multiprocessing import get_context
    root = args.root.rstrip("/")
    mox().file.exists(root)                            # 先单线程初始化 moxing
    os.makedirs(args.local_tmp, exist_ok=True)
    splits = sorted(x.rstrip("/").split("/")[-1] for x in mox().file.list_directory(root))
    print(f"[probe] {root}: {splits}")
    for split in splits:
        d = f"{root}/{split}"
        if not mox().file.is_directory(d):
            continue
        names = sorted(x.rstrip("/").split("/")[-1] for x in mox().file.list_directory(d))
        chunks = [n for n in names if n.endswith(".torch")]
        with ThreadPoolExecutor(32) as ex:
            sizes = list(ex.map(lambda n: mox().file.get_size(f"{d}/{n}"), chunks))
        others = [n for n in names if not n.endswith(".torch")]
        print(f"\n[{split}] {len(chunks)} chunks, {human(sum(sizes))} (chunk min {human(min(sizes))} median {human(float(np.median(sizes)))} "
              f"max {human(max(sizes))}); other files {others[:10]}")
        idx = {}
        if "index.json" in names:
            idx = json.loads(mox().file.read(f"{d}/index.json"))
            ref = set(idx.values())
            print(f"  index.json: {len(idx)} clips in {len(ref)} chunks; chunks referenced but missing {len(ref - set(chunks))}, "
                  f"chunks not referenced {len(set(chunks) - ref)}")
        pick = chunks if args.chunks <= 0 else sorted(random.sample(chunks, min(args.chunks, len(chunks))))
        # 第一个 chunk 详细打印
        local0 = osp.join(args.local_tmp, f"peek_{split}.torch")
        mox().file.copy(f"{d}/{pick[0]}", local0)
        try:
            peek_torch(local0, args.n)
        finally:
            os.remove(local0)
        t0 = time.time()
        rows, errs, n_bytes = [], [], 0
        with get_context("fork").Pool(args.procs) as pool:
            for i, (src, size, r, err, dt) in enumerate(pool.imap_unordered(chunk_stats, [(f"{d}/{c}", args.local_tmp) for c in pick]), 1):
                rows += r
                n_bytes += size
                if err:
                    errs.append((src, err))
                    print(f"  [fail] {src}: {err}", flush=True)
                if i % 50 == 0 or i == len(pick):
                    print(f"  ... {i}/{len(pick)} chunks, {len(rows)} clips, {time.time() - t0:.0f}s", flush=True)
        if not rows:
            continue
        nf = np.array([r["n"] for r in rows])
        dur = np.array([r["dur"] for r in rows])
        whs = Counter(r["wh"] for r in rows)
        fx = np.array([r["fx"] for r in rows if r["fx"] is not None])
        exact = len(pick) == len(chunks)
        scale = 1.0 if exact else len(chunks) / len(pick)
        print(f"  [{split} stats over {len(pick)}/{len(chunks)} chunks{'' if exact else ', 全量为外推'}] "
              f"clips {len(rows)}{'' if exact else f' -> ~{int(len(rows) * scale)}'}, frames {int(nf.sum())}{'' if exact else f' -> ~{int(nf.sum() * scale)}'}")
        print(f"    frames/clip min {nf.min()} p10 {int(np.percentile(nf, 10))} median {int(np.median(nf))} p90 {int(np.percentile(nf, 90))} max {nf.max()}; "
              f"clip 时长 median {np.median(dur):.1f}s max {dur.max():.1f}s")
        print(f"    resolution (w,h): {dict(whs.most_common(5))}; 归一化 fx median {np.median(fx):.3f} (p5 {np.percentile(fx, 5):.3f}, p95 {np.percentile(fx, 95):.3f}); "
              f"jpg 平均 {human(np.mean([r['jpg_bytes'] for r in rows]))}/帧")
        bad = [r["key"] for r in rows if r["bad"]]
        print(f"    异常 clip（NaN / R 非正交 / 帧数不一致）{len(bad)}: {bad[:5]}; 读失败 chunk {len(errs)}")
        if idx:
            miss = [r["key"] for r in rows if r["key"] not in idx]
            print(f"    clip key 不在 index.json 里的 {len(miss)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="/efs-mi-pay-gy1/n84398917/RE10K_HF/train")
    ap.add_argument("--n", type=int, default=2, help="每种后缀抽样解析几个文件")
    ap.add_argument("--workers", type=int, default=32, help="全量扫描的线程数")
    ap.add_argument("--no-scan", action="store_true", help="跳过全量扫描")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=20, help="OBS 模式：每个 split 随机抽几个 chunk 统计；<=0 为全部")
    ap.add_argument("--procs", type=int, default=8, help="OBS 模式：并行下载解析的进程数")
    ap.add_argument("--local-tmp", default="/cache/probe_re10k", help="OBS 模式：chunk 本地临时目录（解析完即删）")
    args = ap.parse_args()
    random.seed(args.seed)
    if args.root.startswith("obs://"):
        return obs_probe(args)
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
