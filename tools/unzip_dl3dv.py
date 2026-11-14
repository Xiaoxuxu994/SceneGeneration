#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 OBS 上 DL3DV 官方发布的逐场景 zip（DL3DV-ALL-960P / 480P / 2K）解压回 OBS，内容原样不改。

    源：  <src>/<批次>/<hash>.zip，zip 内为 <hash>/images_4/frame_xxxxx.png + <hash>/transforms.json
    输出：<out>/<批次>/<hash>/images_4/frame_xxxxx.png
          <out>/<批次>/<hash>/transforms.json      最后写（同步），作为该场景解压完成的标记
          <out>/_index/<shard>.json                本节点完成的场景 {"<批次>/<hash>": 帧数}
    （去掉 zip 里多出的一层 <hash>/；zip 名带 _2 等后缀的（<hash>_2.zip）输出到去掉后缀的 <hash>/）

transforms.json 原样保留：内参是按 3840x2160 原图给的（960P 的图要乘 1/4），transform_matrix 是 nerfstudio 的
c2w（OpenGL 轴向），转换留给读取方。

断点续跑：<out>/<批次>/<hash>/transforms.json 已存在的场景跳过。多节点：--shard i/n 按场景切分。
OBS 小文件写入单节点约 300 个/s 封顶（2026-11-11 实测），960P 约 310 万个文件，单节点约 3 小时，多节点近似线性加速。

    python tools/unzip_dl3dv.py --limit 2                      # 试跑两个场景
    python tools/unzip_dl3dv.py --procs 8 --shard 0/4
    python tools/unzip_dl3dv.py --src obs://bucket-471/dataset/DL3DV-ALL-2K --out .../DL3DV-ALL-2K_to960 \
        --resize-long 960 --img-dir images_4     # 2K 解压并缩到 960x540，目录名与 960P 版一致
"""
import argparse
import io
import json
import os
import re
import os.path as osp
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

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


def convert(out, rel, data):
    """--resize-long > 0 时把图像缩到长边 N（INTER_AREA，宽高比不变）并改放到 --img-dir；其余文件原样。"""
    if not (ARGS.resize_long > 0 and rel.lower().endswith((".png", ".jpg", ".jpeg"))):
        return f"{out}/{rel}", data
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"cannot decode {rel} ({len(data)} bytes)")
    h, w = im.shape[:2]
    s = ARGS.resize_long / max(h, w)
    if s < 1:
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    name = osp.splitext(osp.basename(rel))[0] + ARGS.img_ext
    if ARGS.img_ext == ".jpg":
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok, buf = cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    sub = ARGS.img_dir or osp.dirname(rel)
    return f"{out}/{sub}/{name}", buf.tobytes()


def extract(src, batch, zname, out):
    """下载 <src>/<batch>/<zname>.zip 到本地、解压（可缩放）写到 out；出错抛异常，成功返回 (图像数, 下载耗时)。"""
    t0 = time.time()
    # 先用 mox.file.copy 下载到本地盘再解：mox.file.read 整块读大文件（2K 每个约 1.1GB）很慢，且要全部放进内存
    local = osp.join(ARGS.local_tmp, f"{batch}_{zname}_{osp.basename(src)}.zip")
    os.makedirs(ARGS.local_tmp, exist_ok=True)
    try:
        retry(lambda a, b: mox().file.copy(a, b), f"{src}/{batch}/{zname}.zip", local)
        t_dl = time.time() - t0
        with zipfile.ZipFile(local) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            # 场景根目录 = 最浅的 transforms.json 所在目录（多数 zip 是 <hash>/，少数多套一层如 <hash>/<hash>/）
            tjs = sorted((i for i in infos if osp.basename(i.filename) == "transforms.json"), key=lambda i: i.filename.count("/"))
            if not tjs:
                top = sorted({i.filename.split("/")[0] for i in infos})[:3]
                raise ValueError(f"no transforms.json in zip ({len(infos)} files, top {top})")
            root = osp.dirname(tjs[0].filename)
            root = root + "/" if root else ""
            tj = z.read(tjs[0])
            files = [(i.filename[len(root):], i) for i in infos if i.filename.startswith(root) and i is not tjs[0]]
            with ThreadPoolExecutor(ARGS.threads) as ex:
                list(ex.map(lambda f: write(*convert(out, f[0], z.read(f[1]))), files))
        write(f"{out}/transforms.json", tj)            # 最后写：完成标记
        return sum(r.lower().endswith((".png", ".jpg")) for r, _ in files), t_dl
    finally:
        if osp.exists(local):
            os.remove(local)


def unzip_scene(job):
    batch, zname = job
    h = re.sub(r"_\d+$", "", zname)                   # 部分 zip 名为 <hash>_2.zip（960P 有 955 个，没有同名的 <hash>.zip），场景名去掉后缀
    out = f"{ARGS.out}/{batch}/{h}"
    t0 = time.time()
    try:
        if not ARGS.overwrite and mox().file.exists(f"{out}/transforms.json"):
            n = len(json.loads(mox().file.read(f"{out}/transforms.json")).get("frames", []))
            return batch, h, "skip", n, 0.0
    except Exception as e:
        return batch, h, f"error {type(e).__name__}: {str(e)[:150]}", 0, time.time() - t0
    errs = []
    for src in [ARGS.src] + ARGS.fallback_src:
        try:
            if src == ARGS.src:
                zn = zname
            else:                                      # 备用版本里同一场景的 zip 名可能带 _N 后缀
                cands = [z[:-4] for z in ls(f"{src}/{batch}") if z.endswith(".zip") and re.sub(r"_\d+$", "", z[:-4]) == h]
                if not cands:
                    errs.append(f"{osp.basename(src)}: not found")
                    continue
                zn = cands[0]
            n_img, t_dl = extract(src, batch, zn, out)
            dt = time.time() - t0
            tag = "" if src == ARGS.src else f" from {osp.basename(src)} (fallback; {'; '.join(errs)})"
            return batch, h, f"ok{tag} (download {t_dl:.0f}s process+upload {dt - t_dl:.0f}s)", n_img, dt
        except Exception as e:
            errs.append(f"{osp.basename(src)}: {type(e).__name__}: {str(e)[:150]}")
    return batch, h, "error " + " | ".join(errs), 0, time.time() - t0


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="obs://bucket-471/dataset/DL3DV-ALL-960P")
    ap.add_argument("--out", default="obs://bucket-471/z00533370/dataset/DL3DV-ALL-960P_unzip")
    ap.add_argument("--fallback-src", nargs="*", default=[],
                    help="主版本 zip 坏 / 缺 transforms.json 时依次改用这些版本的同名场景 zip（如 2K 坏包用 DL3DV-ALL-960P 补，图像同为 960x540）")
    ap.add_argument("--batches", nargs="*", default=None, help="默认全部 <N>K 批次")
    ap.add_argument("--procs", type=int, default=16, help="同时解压几个场景（zip 先下载到 --local-tmp，内存占用小）")
    ap.add_argument("--local-tmp", default="/cache/unzip_dl3dv", help="zip 的本地临时目录，解完即删（procs 个 zip 同时存在）")
    ap.add_argument("--threads", type=int, default=16, help="每个场景并发写 OBS 的文件数")
    ap.add_argument("--shard", default="0/1", help="i/n：多节点时本节点处理第 i 份")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只做前 N 个场景（试跑）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resize-long", type=int, default=0,
                    help=">0 时把图像缩到长边 N 再存（如 2K 的 1920x1080 -> 960x540）；transforms.json 仍原样（内参按原图 3840x2160）")
    ap.add_argument("--img-dir", default=None, help="缩放后的图像目录名（默认沿用 zip 里的，如 images_2）；2K 缩到 960 时用 images_4 与 960P 版一致")
    ap.add_argument("--img-ext", default=".png", choices=[".png", ".jpg"], help="缩放后的图像格式（jpg 为 q95，体积约 png 的 1/4）")
    ARGS = ap.parse_args()

    mox().file.exists(ARGS.src)                        # 先单线程初始化 moxing
    batches = ARGS.batches or sorted((b for b in ls(ARGS.src) if b.endswith("K") and b[:-1].isdigit()), key=lambda b: int(b[:-1]))
    jobs = [(b, z[:-4]) for b in batches for z in ls(f"{ARGS.src}/{b}") if z.endswith(".zip")]
    si, sn = (int(x) for x in ARGS.shard.split("/"))
    jobs = jobs[si::sn]
    if ARGS.limit:
        jobs = jobs[: ARGS.limit]
    print(f"[DL3DV] {ARGS.src} -> {ARGS.out}  batches {batches}  shard {ARGS.shard}: {len(jobs)} scenes, "
          f"procs={ARGS.procs} threads={ARGS.threads} resize_long={ARGS.resize_long} img_dir={ARGS.img_dir} img_ext={ARGS.img_ext}", flush=True)

    t0, done, n_err, n_img, index = time.time(), 0, 0, 0, {}
    with get_context("fork").Pool(ARGS.procs) as pool:
        for batch, h, status, n, dt in pool.imap_unordered(unzip_scene, jobs):
            done += 1
            if status.startswith("error"):
                n_err += 1
                print(f"  {batch}/{h}: {status}", flush=True)
            else:
                index[f"{batch}/{h}"] = n
                n_img += n
                if status.startswith("ok") and (done <= 10 or "fallback" in status):          # 前几个场景打出分阶段耗时，便于判断瓶颈
                    print(f"  {batch}/{h}: {n} frames, {status}", flush=True)
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[DL3DV] {done}/{len(jobs)} scenes ({n_err} errors), {n_img} frames, {el / 60:.1f} min, "
                      f"eta {el / done * (len(jobs) - done) / 60:.1f} min", flush=True)
    if not ARGS.limit:
        tag = ARGS.shard.replace("/", "of")
        write(f"{ARGS.out}/_index/{tag}.json", json.dumps(index).encode())
    print(f"[DL3DV] finished: {len(index)} scenes, {n_img} frames, {n_err} errors（重跑同一条命令只补做没完成的场景）", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
