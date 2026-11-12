#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只看目录结构地探查一个 OBS（或本地）数据集目录，不遍历全部文件：
  - 根目录：条目数、前后几个名字、按前缀（如 1K / 2K ...）计数；
  - 抽样 / 指定几个场景：逐层列目录（每层只列一次，最多 --depth 层），给出每个子目录的文件数、后缀统计和前几个文件名；
  - 场景里的小 json（<= --max-json-mb）：打印顶层 key、列表长度和第一个元素，用来看标签格式；
  - 可选 --peek：读每种后缀的第一个文件，打印图像 / npy / exr / h5 的形状、dtype 和取值范围；
  - 可选 --siblings：列出根目录的上一级，找同级的其他数据（如 RGB 原图目录）。

    python tools/probe_obs_layout.py obs://bucket-471/y00959808/data/sr_datasets/processed/dl3dv \
        --scenes 5K-a85c949d27a67ebded3ed23a20a911fd6a751c1eae945dc4d10d0afe442e1294 --n 2 --peek --siblings
"""
import argparse
import io
import json
import os.path as osp
import random
import re
import sys
from collections import Counter

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def mox():
    import moxing
    return moxing


def is_obs(p):
    return p.startswith("obs://")


def ls(p):
    if is_obs(p):
        return sorted(x.rstrip("/").split("/")[-1] for x in mox().file.list_directory(p))
    import os
    return sorted(os.listdir(p))


def isdir(p):
    if is_obs(p):
        return mox().file.is_directory(p)
    return osp.isdir(p)


def read(p):
    if is_obs(p):
        return mox().file.read(p, binary=True)
    return open(p, "rb").read()


def size(p):
    if is_obs(p):
        return mox().file.get_size(p)
    return osp.getsize(p)


def ext_of(name):
    m = re.search(r"(\.[A-Za-z0-9]+)+$", name)
    return m.group(0).lower() if m else "(none)"


def describe_json(data, indent):
    try:
        obj = json.loads(data)
    except Exception as e:
        print(f"{indent}json parse failed: {e}")
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:40]:
            if isinstance(v, list):
                print(f"{indent}{k}: list[{len(v)}]" + (f" first={json.dumps(v[0])[:400]}" if v else ""))
            elif isinstance(v, dict):
                print(f"{indent}{k}: dict keys={list(v)[:15]}")
            else:
                print(f"{indent}{k}: {json.dumps(v)[:200]}")
    else:
        print(f"{indent}{type(obj).__name__} len={len(obj)}: {json.dumps(obj)[:400]}")


def peek(path, name, data, indent):
    import numpy as np
    e = ext_of(name)
    try:
        if e.endswith((".png", ".jpg", ".jpeg")):
            import cv2
            a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        elif e.endswith(".npy"):
            a = np.load(io.BytesIO(data), allow_pickle=False)
        elif e.endswith(".npz"):
            z = np.load(io.BytesIO(data))
            print(f"{indent}{name}: npz " + ", ".join(f"{k}:{z[k].dtype}{z[k].shape}" for k in z.files))
            return
        elif e.endswith(".exr"):
            import os
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            import cv2
            a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        elif e.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(io.BytesIO(data), "r") as f:
                print(f"{indent}{name}: h5 " + ", ".join(f"{k}:{getattr(f[k], 'shape', '')}" for k in list(f.keys())[:10]))
            return
        else:
            return
        if a is None:
            print(f"{indent}{name}: decode failed")
            return
        a = np.asarray(a)
        fin = a[np.isfinite(a)] if a.dtype.kind == "f" else a
        print(f"{indent}{name}: {a.dtype} {a.shape} min={fin.min() if fin.size else 'nan'} max={fin.max() if fin.size else 'nan'}"
              + (f" zero%={(a == 0).mean():.1%}" if a.dtype.kind == "f" else ""))
    except Exception as ex:
        print(f"{indent}{name}: peek failed ({type(ex).__name__}: {str(ex)[:120]})")


def walk(path, depth, args, indent="  "):
    names = ls(path)
    exts = Counter(ext_of(n) for n in names)
    print(f"{indent}{path.split('/')[-1]}/  {len(names)} entries, by ext: {dict(exts.most_common(8))}")
    print(f"{indent}  first: {names[:args.max_list]}" + (f"  last: {names[-3:]}" if len(names) > args.max_list else ""))
    seen_ext = set()
    for n in names:
        p = f"{path}/{n}"
        e = ext_of(n)
        if e == "(none)" and depth > 1 and isdir(p):
            walk(p, depth - 1, args, indent + "    ")
            continue
        if e.endswith(".json") and size(p) <= args.max_json_mb * 1e6 and n not in seen_ext:
            print(f"{indent}  [{n}] ({size(p) / 1e3:.1f} KB)")
            describe_json(read(p), indent + "    ")
            seen_ext.add(n)
        elif args.peek and e not in seen_ext and e != "(none)":
            peek(p, n, read(p), indent + "  ")
            seen_ext.add(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--scenes", nargs="*", default=[], help="一定要看的场景名")
    ap.add_argument("--n", type=int, default=2, help="另外随机抽几个场景")
    ap.add_argument("--depth", type=int, default=3, help="场景内往下列几层")
    ap.add_argument("--max-list", type=int, default=8)
    ap.add_argument("--max-json-mb", type=float, default=5)
    ap.add_argument("--peek", action="store_true", help="每种后缀读一个文件看形状 / dtype / 范围")
    ap.add_argument("--siblings", action="store_true", help="列出根目录的上一级")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.root.rstrip("/")
    if args.siblings:
        parent = root.rsplit("/", 1)[0]
        print(f"[parent] {parent}: {ls(parent)}", flush=True)
    names = ls(root)
    print(f"[root] {root}: {len(names)} entries", flush=True)
    print(f"  first: {names[:10]}\n  last:  {names[-5:]}")
    prefixes = Counter(n.split("-")[0] for n in names if "-" in n)
    if prefixes:
        print(f"  by prefix before '-': {dict(sorted(prefixes.items()))}")
    files = [n for n in names if ext_of(n) != "(none)"]
    if files:
        print(f"  files at root: {files[:20]}")
        for n in files:
            if n.endswith(".json") and size(f"{root}/{n}") <= args.max_json_mb * 1e6:
                print(f"  [{n}]")
                describe_json(read(f"{root}/{n}"), "    ")

    dirs = [n for n in names if ext_of(n) == "(none)"]
    pick = [s for s in args.scenes if s in names]
    for s in args.scenes:
        if s not in names:
            print(f"[WARN] scene {s} not under {root}")
    random.Random(args.seed).shuffle(dirs)
    pick += [d for d in dirs if d not in pick][: args.n]
    for s in pick:
        print(f"\n[scene] {s}", flush=True)
        walk(f"{root}/{s}", args.depth, args)


if __name__ == "__main__":
    main()
