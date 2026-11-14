#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 OBS 上 pixelSplat 格式的 RealEstate10K（<src>/<split>/xxxxxx.torch，每个 chunk 约 100MB、十几个视频切片）
转成每个切片一个 tar，供打标和训练按场景随机读取：

    <out>/<split>/<key>.tar
        images/000000.jpg ...       chunk 里的原始图像码流，不解码不重压（后缀按码流判断，jpg / png）
        meta.json                   key, url, split, chunk, width, height, frames（图像文件名）, timestamps（微秒）,
                                    cameras_raw（N x 18 原样：fx fy cx cy 归一化 + 2 个 0 + 3x4 w2c）,
                                    intrinsics（N x 4 像素单位 fx fy cx cy = 归一化值 x 宽 / 高）,
                                    w2c（N x 3 x 4，OpenCV 轴向，与 cameras_raw[:, 6:] 相同）
    <out>/<split>/_done/<chunk>.json    该 chunk 的切片全部写完后写（断点续跑标记）：{key: [帧数, 宽, 高]}
    <out>/<split>/_index.json           所有 chunk 做完后合并：{key: [帧数, 宽, 高]}

位姿来自 SLAM，尺度不确定，没有深度。全部帧保留不抽帧（抽帧留给训练采样 / 打标）。
断点续跑：_done/<chunk>.json 已存在的 chunk 跳过。多节点：--shard i/n 按 chunk 切分；
每个节点结束时若全部 chunk 都已完成就合并 _index.json（也可单独 --merge-only）。

    python tools/re10k_to_tar.py --limit 2                 # 试跑：每个 split 只做前 2 个 chunk
    python tools/re10k_to_tar.py --procs 16 --shard 0/2
    python tools/re10k_to_tar.py --merge-only
"""
import argparse
import io
import json
import os
import os.path as osp
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np

ARGS = None


def mox():
    import moxing
    return moxing


def ls(p):
    return sorted(x.rstrip("/").split("/")[-1] for x in mox().file.list_directory(p))


def retry(fn, *a, n=3):
    err = None
    for i in range(n):
        try:
            return fn(*a)
        except Exception as e:
            err = e
            time.sleep(1 + i)
    raise err


def write(path, data):
    retry(lambda p, d: mox().file.write(p, d, binary=True), path, data)


def img_ext(b):
    return ".png" if b[:4] == b"\x89PNG" else ".jpg"


def clip_tar(x, split, chunk):
    """一个视频切片 -> (tar 字节, 帧数, 宽, 高)。"""
    from PIL import Image
    ims = [im.numpy().tobytes() if hasattr(im, "numpy") else bytes(im) for im in x["images"]]
    cams = np.asarray(x["cameras"], dtype=np.float64).reshape(-1, 18)
    ts = [int(t) for t in np.asarray(x["timestamps"]).reshape(-1)]
    assert len(ims) == len(cams) == len(ts), f"length mismatch images {len(ims)} cameras {len(cams)} timestamps {len(ts)}"
    w, h = Image.open(io.BytesIO(ims[0])).size
    names = [f"images/{i:06d}{img_ext(b)}" for i, b in enumerate(ims)]
    k = cams[:, :4] * np.array([w, h, w, h])
    meta = dict(key=x["key"], url=x.get("url"), split=split, chunk=chunk, width=w, height=h, frames=names, timestamps=ts,
                cameras_raw=cams.tolist(), intrinsics=k.round(4).tolist(), w2c=cams[:, 6:].reshape(-1, 3, 4).tolist())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, b in zip(names + ["meta.json"], ims + [json.dumps(meta).encode()]):
            info = tarfile.TarInfo(name)
            info.size = len(b)
            tar.addfile(info, io.BytesIO(b))
    return buf.getvalue(), len(ims), w, h


def do_chunk(job):
    split, chunk = job
    out = f"{ARGS.out}/{split}"
    done = f"{out}/_done/{chunk[:-6]}.json"
    t0 = time.time()
    local = osp.join(ARGS.local_tmp, f"{split}_{chunk}")
    try:
        if not ARGS.overwrite and mox().file.exists(done):
            d = json.loads(mox().file.read(done))
            return split, chunk, "skip", d, 0.0
        import torch
        os.makedirs(ARGS.local_tmp, exist_ok=True)
        retry(lambda a, b: mox().file.copy(a, b), f"{ARGS.src}/{split}/{chunk}", local)
        t_dl = time.time() - t0
        items = torch.load(local, map_location="cpu", weights_only=False)
        os.remove(local)
        res, bad = {}, []

        def one(x):
            try:
                data, n, w, h = clip_tar(x, split, chunk)
                write(f"{out}/{x['key']}.tar", data)
                return x["key"], [n, w, h], None
            except Exception as e:
                return x.get("key", "?"), None, f"{type(e).__name__}: {str(e)[:120]}"

        with ThreadPoolExecutor(ARGS.threads) as ex:
            for key, v, err in ex.map(one, items):
                if err:
                    bad.append(f"{key}: {err}")
                else:
                    res[key] = v
        if bad:                                        # 有切片失败就不写完成标记，重跑时整块重做
            return split, chunk, "error " + "; ".join(bad[:3]), res, time.time() - t0
        write(done, json.dumps(res).encode())
        dt = time.time() - t0
        return split, chunk, f"ok (download {t_dl:.0f}s convert+upload {dt - t_dl:.0f}s)", res, dt
    except Exception as e:
        if osp.exists(local):
            os.remove(local)
        return split, chunk, f"error {type(e).__name__}: {str(e)[:150]}", {}, time.time() - t0


def merge_index(split, n_chunks):
    d = f"{ARGS.out}/{split}/_done"
    names = [n for n in ls(d) if n.endswith(".json")] if mox().file.exists(d) else []
    if len(names) < n_chunks:
        print(f"[RE10K] {split}: {len(names)}/{n_chunks} chunks done, _index.json not merged yet", flush=True)
        return
    index = {}
    with ThreadPoolExecutor(32) as ex:
        for part in ex.map(lambda n: json.loads(mox().file.read(f"{d}/{n}")), names):
            index.update(part)
    write(f"{ARGS.out}/{split}/_index.json", json.dumps(index).encode())
    nf = np.array([v[0] for v in index.values()])
    print(f"[RE10K] {split}: _index.json {len(index)} clips, {int(nf.sum())} frames "
          f"(frames/clip min {nf.min()} median {int(np.median(nf))} max {nf.max()})", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="obs://bucket-471/dataset/re10k")
    ap.add_argument("--out", default="obs://bucket-471/z00533370/dataset/RE10K_tar")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--procs", type=int, default=16, help="同时处理几个 chunk（每个先下载到 --local-tmp，约 100MB）")
    ap.add_argument("--threads", type=int, default=8, help="每个 chunk 并发写 OBS 的 tar 数")
    ap.add_argument("--local-tmp", default="/cache/re10k_to_tar")
    ap.add_argument("--shard", default="0/1", help="i/n：多节点时本节点处理第 i 份 chunk")
    ap.add_argument("--limit", type=int, default=0, help=">0 时每个 split 只做前 N 个 chunk（试跑，不合并索引）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="只合并 _done/ 为 _index.json")
    ARGS = ap.parse_args()

    mox().file.exists(ARGS.src)                        # 先单线程初始化 moxing
    chunks = {s: [c for c in ls(f"{ARGS.src}/{s}") if c.endswith(".torch")] for s in ARGS.splits}
    if ARGS.merge_only:
        for s in ARGS.splits:
            merge_index(s, len(chunks[s]))
        return 0
    jobs = []
    for s in ARGS.splits:
        cs = chunks[s][: ARGS.limit] if ARGS.limit else chunks[s]
        jobs += [(s, c) for c in cs]
    si, sn = (int(x) for x in ARGS.shard.split("/"))
    jobs = jobs[si::sn]
    print(f"[RE10K] {ARGS.src} -> {ARGS.out}  " + ", ".join(f"{s} {len(c)} chunks" for s, c in chunks.items())
          + f"  shard {ARGS.shard}: {len(jobs)} chunks, procs={ARGS.procs} threads={ARGS.threads}", flush=True)

    t0, done, n_err, n_clip, n_frame = time.time(), 0, 0, 0, 0
    with get_context("fork").Pool(ARGS.procs) as pool:
        for split, chunk, status, res, dt in pool.imap_unordered(do_chunk, jobs):
            done += 1
            if status.startswith("error"):
                n_err += 1
                print(f"  {split}/{chunk}: {status}", flush=True)
            else:
                n_clip += len(res)
                n_frame += sum(v[0] for v in res.values())
                if status.startswith("ok") and done <= 5:
                    print(f"  {split}/{chunk}: {len(res)} clips, {status}", flush=True)
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[RE10K] {done}/{len(jobs)} chunks ({n_err} errors), {n_clip} clips, {n_frame} frames, {el / 60:.1f} min, "
                      f"eta {el / done * (len(jobs) - done) / 60:.1f} min", flush=True)
    if not ARGS.limit:
        for s in ARGS.splits:
            merge_index(s, len(chunks[s]))
    print(f"[RE10K] finished: {n_clip} clips, {n_frame} frames, {n_err} chunk errors（重跑同一条命令只补做没完成的 chunk）", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
