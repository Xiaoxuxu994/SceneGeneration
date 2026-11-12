#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL3DV（obs://bucket-471/y00959808/data/sr_datasets/processed/dl3dv，布局 <1K..7K>/<hash>/）的场景数、RGB、相机检查。
不看深度。逐个场景只列目录（多线程），不读帧文件：
  1) 全量：每个场景的子目录、images_8 帧数、场景根目录的文件（相机文件如 transforms.json）-> <out>/dl3dv_scenes.csv
  2) 相机来源：manifest_*.jsonl（只覆盖部分场景）、meta/<批次>.json、pkl_map/meta_data.pkl、sequence_list.pkl 各覆盖多少场景
  3) 抽样（含 --scenes 指定的）：读第一帧 png 的尺寸，和相机文件里的内参 / 宽高对照，并检查帧名与相机帧是否一一对应

    python tools/check_dl3dv_obs.py --scenes 5K/a85c949d27a67ebded3ed23a20a911fd6a751c1eae945dc4d10d0afe442e1294 --n 3
"""
import argparse
import csv
import io
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "obs://bucket-471/y00959808/data/sr_datasets/processed/dl3dv"
RGB_DIR = "images_8"


def mox():
    import moxing
    return moxing


def ls(p):
    return sorted(x.rstrip("/").split("/")[-1] for x in mox().file.list_directory(p))


def scan_scene(batch, h):
    d = f"{ROOT}/{batch}/{h}"
    try:
        names = ls(d)
        n_rgb = len(ls(f"{d}/{RGB_DIR}")) if RGB_DIR in names else 0
        return dict(batch=batch, hash=h, n_rgb=n_rgb, entries=";".join(names), err="")
    except Exception as e:
        return dict(batch=batch, hash=h, n_rgb=0, entries="", err=f"{type(e).__name__}: {str(e)[:100]}")


def scene_ids(obj):
    """从各种索引结构里取出场景 hash 集合（dict 的 key、list 里的字符串或含 scene_id 的条目）。"""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and len(k) >= 32:
                out.add(k.split("/")[-1].split("-")[-1])
            elif isinstance(v, (dict, list)):
                out |= scene_ids(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, str) and len(v) >= 32:
                out.add(v.rstrip("/").split("/")[-1].split("-")[-1])
            elif isinstance(v, dict) and "scene_id" in v:
                out.add(str(v["scene_id"]).split("/")[-1].split("-")[-1])
            elif isinstance(v, (dict, list)):
                out |= scene_ids(v)
    return out


def brief(obj, n=600):
    if isinstance(obj, dict):
        k = next(iter(obj))
        return f"dict len={len(obj)}; first key={k!r}; value={json.dumps(obj[k], default=str)[:n]}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} len={len(obj)}; first={json.dumps(obj[0], default=str)[:n] if obj else None}"
    return f"{type(obj).__name__}: {str(obj)[:n]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=[], help="一定要抽查的场景，写成 <批次>/<hash>")
    ap.add_argument("--n", type=int, default=3, help="另外随机抽查几个场景")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default="/efs-mi-pay-gy1/z00533370/exp/logs/dl3dv_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) 全量列目录
    t0 = time.time()
    batches = [b for b in ls(ROOT) if b.endswith("K") and b[:-1].isdigit()]
    jobs = [(b, h) for b in batches for h in ls(f"{ROOT}/{b}")]
    print(f"[scan] {len(batches)} batches {batches}, {len(jobs)} scene dirs; listing each ...", flush=True)
    with ThreadPoolExecutor(args.threads) as ex:
        rows = list(ex.map(lambda j: scan_scene(*j), jobs))
    with open(f"{args.out}/dl3dv_scenes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_rgb = np.array([r["n_rgb"] for r in rows])
    entry_cnt = Counter(e for r in rows for e in r["entries"].split(";") if e)
    print(f"[scan] done in {time.time() - t0:.0f}s -> {args.out}/dl3dv_scenes.csv", flush=True)
    print(f"  per batch: {dict(Counter(r['batch'] for r in rows))}")
    print(f"  scenes with {RGB_DIR}: {(n_rgb > 0).sum()}/{len(rows)}; frames total {n_rgb.sum()}, "
          f"per scene min/median/max {n_rgb[n_rgb > 0].min() if (n_rgb > 0).any() else 0}/"
          f"{int(np.median(n_rgb[n_rgb > 0])) if (n_rgb > 0).any() else 0}/{n_rgb.max()}")
    print(f"  entries in scene dirs (how many scenes have each): {dict(entry_cnt.most_common(20))}")
    errs = [r for r in rows if r["err"]]
    if errs:
        print(f"  list errors: {len(errs)}, e.g. {errs[:2]}")
    all_hash = {r["hash"] for r in rows}

    # 2) 相机 / 索引来源各覆盖多少场景
    print("\n[index files]", flush=True)
    sources = {}
    for name in ["manifest_train.jsonl", "manifest_eval.jsonl"]:
        lines = mox().file.read(f"{ROOT}/{name}").splitlines()
        sources[name] = {json.loads(l)["scene_id"] for l in lines if l.strip()}
    for name in ["sequence_list.pkl", "pkl_map/meta_data.pkl"]:
        try:
            obj = pickle.loads(mox().file.read(f"{ROOT}/{name}", binary=True))
            print(f"  {name}: {brief(obj)}", flush=True)
            sources[name] = scene_ids(obj)
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {str(e)[:150]}")
    for b in batches:
        p = f"{ROOT}/meta/{b}.json"
        try:
            obj = json.loads(mox().file.read(p))
            if b == batches[0]:
                print(f"  meta/{b}.json ({mox().file.get_size(p) / 1e6:.1f} MB): {brief(obj)}", flush=True)
            sources[f"meta/{b}.json"] = scene_ids(obj)
        except Exception as e:
            print(f"  meta/{b}.json: {type(e).__name__}: {str(e)[:150]}")
    for name, ids in sources.items():
        print(f"  {name:28s} {len(ids):6d} ids, {len(ids & all_hash):6d} match scene dirs")
    meta_all = set().union(*[v for k, v in sources.items() if k.startswith("meta/")]) if any(k.startswith("meta/") for k in sources) else set()
    print(f"  scenes with {RGB_DIR} AND an entry in meta/*.json: "
          f"{len({r['hash'] for r in rows if r['n_rgb'] > 0} & meta_all)}")

    # 3) 抽查：图像尺寸 vs 相机
    import cv2
    pick = list(args.scenes)
    rest = [f"{r['batch']}/{r['hash']}" for r in rows if r["n_rgb"] > 0]
    random.Random(0).shuffle(rest)
    pick += [s for s in rest if s not in pick][: args.n]
    for s in pick:
        d = f"{ROOT}/{s}"
        print(f"\n[sample] {s}", flush=True)
        try:
            names = ls(d)
            print(f"  entries: {names}")
            frames = ls(f"{d}/{RGB_DIR}") if RGB_DIR in names else []
            if frames:
                im = cv2.imdecode(np.frombuffer(mox().file.read(f"{d}/{RGB_DIR}/{frames[0]}", binary=True), np.uint8), cv2.IMREAD_UNCHANGED)
                print(f"  {RGB_DIR}: {len(frames)} frames, {frames[:2]} ... {frames[-1:]}, image {im.shape} {im.dtype}")
            for n in names:
                if n.endswith(".json"):
                    obj = json.loads(mox().file.read(f"{d}/{n}"))
                    keys = {k: (v if not isinstance(v, list) else f"list[{len(v)}]") for k, v in obj.items()} if isinstance(obj, dict) else type(obj).__name__
                    print(f"  {n}: {json.dumps(keys, default=str)[:900]}")
                    fr = obj.get("frames") if isinstance(obj, dict) else None
                    if fr:
                        print(f"    frame[0]: {json.dumps(fr[0], default=str)[:700]}")
                        cam_names = {os.path.basename(str(x.get('file_path', ''))).rsplit('.', 1)[0] for x in fr}
                        rgb_names = {f.rsplit('.', 1)[0] for f in frames}
                        print(f"    frames in json {len(fr)}, rgb files {len(rgb_names)}, matched by name {len(cam_names & rgb_names)}")
            h = s.split("/")[-1]
            for name in ["manifest_train.jsonl", "manifest_eval.jsonl"]:
                if h in sources.get(name, set()):
                    print(f"  also in {name}")
        except Exception as e:
            print(f"  {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    sys.exit(main())
