#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialVID-HQ（OBS 上的 HF 镜像，videos / annotations / depths 下是 group_XXXX.tar.gz）解压到 OBS。
tar.gz 从 OBS 流式读取、边解边写回 OBS（不落本地盘）；流式读取不可用时加 --via-local 先拷到本地盘再解。

三种用法：
  1) 看结构 + 只解几条（先从 depths 取 n 个 clip，再解同一批 clip 的 annotation / video）：
     python tools/spatialvid_extract.py sample --n 3
  2) 看解出来的样本里有哪些 GT（下载到本地临时目录，逐个文件打印格式、形状、取值范围）：
     python tools/spatialvid_extract.py inspect
  3) 全量（按 group 并行；每个 group / 类别写完记 _progress/<kind>_<group>.done，重跑跳过）：
     python tools/spatialvid_extract.py all --procs 8
     python tools/spatialvid_extract.py all --groups group_0001 group_0002 --kinds annotations depths

输出：<dst>/<kind>/group_XXXX/<clip>/...（tar 内路径去掉 group 之前的 SpatialVid/HQ/<kind>/），
      例如 obs://bucket-471/z00533370/dataset/SpatialVID-HQ/annotations/group_0001/<clip>/poses.npy
"""
import argparse
import io
import json
import os
import os.path as osp
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from datasets.base import fileio as fio  # noqa: E402

SRC = "obs://bucket-471/dataset/SpatialVID/spatialvid-hq"
DST = "obs://bucket-471/z00533370/dataset/SpatialVID-HQ"
KINDS = ("annotations", "depths", "videos")


def mox():
    return fio._mox()


def _parts_from_group(name):
    """tar 成员路径从 group_XXXX 那一级开始的各段（实际成员形如 SpatialVid/HQ/annotations/group_0001/<clip>/poses.npy）。"""
    parts = [p for p in name.split("/") if p and p != "."]
    for i, p in enumerate(parts):
        if p.startswith("group_"):
            return parts[i:]
    return parts


def rel_path(name):
    """写到 <dst>/<kind>/ 下的相对路径：group_XXXX/<clip>/...（去掉 group 之前的 SpatialVid/HQ/<kind>/）。"""
    return "/".join(_parts_from_group(name))


def clip_key(name):
    """tar 成员路径 -> clip id：group_XXXX 下一级的目录名或文件名（去扩展名）。"""
    parts = _parts_from_group(name)
    if parts and parts[0].startswith("group_"):
        parts = parts[1:]
    return parts[0].split(".")[0] if parts else ""


class _Reader(io.RawIOBase):
    """把 mox.file.File 包成 tarfile 流式模式能用的只读流。"""

    def __init__(self, f):
        self.f = f

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def open_tar(path, via_local=None):
    if via_local:
        os.makedirs(via_local, exist_ok=True)
        local = osp.join(via_local, osp.basename(path))
        if not osp.exists(local) or osp.getsize(local) != mox().file.get_size(path):
            t = time.time()
            mox().file.copy(path, local)
            print(f"  copied {path} -> {local} ({time.time() - t:.0f}s)", flush=True)
        return tarfile.open(local, mode="r|gz"), local
    stream = io.BufferedReader(_Reader(mox().file.File(path, "rb")), buffer_size=16 << 20)
    return tarfile.open(fileobj=stream, mode="r|gz"), None


def iter_members(path, via_local=None):
    """(member name, bytes) 逐个产出（只产出普通文件）。"""
    tf, local = open_tar(path, via_local)
    try:
        for m in tf:
            if m.isfile():
                yield m.name, tf.extractfile(m).read()
    finally:
        tf.close()
        if local:
            os.remove(local)


def list_groups(kind):
    names = mox().file.list_directory(f"{SRC}/{kind}")
    return sorted(n for n in names if n.endswith(".tar.gz"))


# ---------------------------------------------------------------------------
def cmd_sample(args):
    print(f"[src] {SRC}", flush=True)
    print("  top:", sorted(mox().file.list_directory(SRC)), flush=True)
    for kind in KINDS + ("data",):
        try:
            names = sorted(mox().file.list_directory(f"{SRC}/{kind}"))
            print(f"  {kind}/: {len(names)} entries, first {names[:4]} last {names[-2:]}", flush=True)
        except Exception as e:
            print(f"  {kind}/: {e}", flush=True)
    for meta in ("README.md", "dataset_infos.json"):
        try:
            fio.write_bytes(f"{DST}/{meta}", fio.read_bytes(f"{SRC}/{meta}"))
        except Exception as e:
            print(f"  copy {meta}: {e}", flush=True)

    group = args.group
    # 1) depths：取前 n 个 clip。depths 只有部分 group（24/74），且 tar 内顺序与 annotations / videos 不同，
    #    所以以有深度的 clip 为准，再去另外两类里找同一批
    keys, t = [], time.time()
    src = f"{SRC}/depths/{group}.tar.gz"
    print(f"\n[depths] {src}", flush=True)
    for name, data in iter_members(src, args.via_local):
        k = clip_key(name)
        if k not in keys:
            if len(keys) >= args.n:
                break
            keys.append(k)
        fio.write_bytes(f"{DST}/depths/{rel_path(name)}", data)
        print(f"  {name}  {len(data) / 1e6:.2f} MB", flush=True)
    print(f"  clips: {keys}  ({time.time() - t:.0f}s)", flush=True)

    # 2) annotations / videos：找同一批 clip（tar 内顺序不同，可能要扫完整个 tar）
    for kind in ("annotations", "videos"):
        src = f"{SRC}/{kind}/{group}.tar.gz"
        print(f"\n[{kind}] {src}", flush=True)
        found, scanned, t = set(), 0, time.time()
        for name, data in iter_members(src, args.via_local):
            scanned += 1
            if scanned <= 3:
                print(f"  (member #{scanned}: {name})", flush=True)
            if scanned % 5000 == 0:
                print(f"  ... scanned {scanned} members, found {len(found)}/{len(keys)} ({time.time() - t:.0f}s)", flush=True)
            k = clip_key(name)
            if k in keys:
                found.add(k)
                fio.write_bytes(f"{DST}/{kind}/{rel_path(name)}", data)
                print(f"  {name}  {len(data) / 1e6:.2f} MB", flush=True)
            elif found == set(keys) or (args.scan_limit and scanned >= args.scan_limit):
                break
        miss = [k for k in keys if k not in found]
        print(f"  found {len(found)}/{len(keys)} clips after {scanned} members ({time.time() - t:.0f}s)"
              + (f"; missing {miss}" if miss else ""), flush=True)
    print(f"\n[done] samples under {DST}; next: python tools/spatialvid_extract.py inspect", flush=True)


# ---------------------------------------------------------------------------
def _describe(path, data):
    import numpy as np
    ext = osp.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            a = np.load(io.BytesIO(data), allow_pickle=True)
            s = f"npy {a.dtype} {a.shape}"
            if a.dtype.kind in "fiu" and a.size:
                s += f" min={a.min():.4g} max={a.max():.4g}"
                s += f"\n        first: {np.array2string(a.reshape(-1, a.shape[-1])[0] if a.ndim > 1 else a[:8], precision=4)}"
            return s
        if ext == ".npz":
            z = np.load(io.BytesIO(data), allow_pickle=True)
            return "npz " + ", ".join(f"{k}:{z[k].dtype}{z[k].shape}" for k in z.files)
        if ext in (".json", ".jsonl"):
            txt = data.decode("utf-8", "replace")
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return "json keys: " + ", ".join(f"{k}({type(v).__name__})" for k, v in list(obj.items())[:20])
                return f"json {type(obj).__name__} len={len(obj)}: {txt[:300]}"
            except Exception:
                return "text: " + txt[:300]
        if ext in (".txt", ".csv"):
            return "text: " + data.decode("utf-8", "replace")[:400]
        if ext in (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"):
            import cv2
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            return f"image {im.dtype} {im.shape} min={im.min()} max={im.max()}" if im is not None else "image (decode failed)"
        if ext == ".zip":
            import zipfile
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = zf.namelist()
            s = f"zip {len(names)} files: {names[:3]} ... {names[-2:]}"
            first = next((n for n in names if not n.endswith("/")), None)
            if first:
                s += f"\n        first member {first}: " + _describe(first, zf.read(first))
            return s
        if ext in (".mp4", ".mkv", ".avi", ".mov"):
            import cv2
            with tempfile.NamedTemporaryFile(suffix=ext) as tf:
                tf.write(data)
                tf.flush()
                cap = cv2.VideoCapture(tf.name)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            return f"video {w}x{h} {n} frames {fps:.2f} fps ({n / max(fps, 1e-6):.1f}s)"
    except Exception as e:
        return f"{ext} (failed to parse: {e})"
    return f"{ext or 'no ext'} {len(data)} bytes"


def cmd_inspect(args):
    try:
        readme = fio.read_bytes(f"{SRC}/README.md").decode("utf-8", "replace")
        keep = [l for l in readme.splitlines()
                if any(w in l.lower() for w in ("depth", "pose", "intrinsic", "extrinsic", "camera", "mask", "fps", "annotation", "c2w", "w2c", "unit", "scale"))]
        print("[README lines about GT]\n  " + "\n  ".join(keep[:60]), flush=True)
    except Exception as e:
        print(f"[README] {e}", flush=True)
    for kind in KINDS:
        root = f"{DST}/{kind}"
        if not mox().file.exists(root):
            print(f"\n[{kind}] not extracted yet ({root})", flush=True)
            continue
        files = sorted(f for f in mox().file.list_directory(root, recursive=True) if not f.endswith("/"))
        keys = []
        for f in files:
            k = clip_key(f)
            if k not in keys:
                keys.append(k)
        print(f"\n[{kind}] {len(files)} files, {len(keys)} clips under {root}", flush=True)
        shown = [f for f in files if clip_key(f) in keys[: args.clips]]
        for f in shown[: args.max_files]:
            print(f"  {f}: {_describe(f, fio.read_bytes(f'{root}/{f}'))}", flush=True)
    try:
        names = sorted(mox().file.list_directory(f"{SRC}/data", recursive=True))
        print(f"\n[data/] {len(names)} entries: {names[:10]}", flush=True)
        for n in [x for x in names if not x.endswith("/")][:3]:
            p = f"{SRC}/data/{n}"
            if not mox().file.is_directory(p) and mox().file.get_size(p) < 200 << 20:
                print(f"  {n}: {_describe(n, fio.read_bytes(p)) if not n.endswith('.parquet') else 'parquet'}", flush=True)
    except Exception as e:
        print(f"[data/] {e}", flush=True)


# ---------------------------------------------------------------------------
def _extract_group(job):
    kind, group, via_local, threads = job
    done = f"{DST}/_progress/{kind}_{group}.done"
    if mox().file.exists(done):
        return kind, group, "skip", 0, 0.0
    t, n, nbytes = time.time(), 0, 0
    with ThreadPoolExecutor(threads) as pool:
        futs = []
        for name, data in iter_members(f"{SRC}/{kind}/{group}.tar.gz", via_local):
            futs.append(pool.submit(fio.write_bytes, f"{DST}/{kind}/{rel_path(name)}", data))
            n += 1
            nbytes += len(data)
            if len(futs) >= 4 * threads:          # 限制内存里待写的文件数
                for f in futs:
                    f.result()
                futs = []
        for f in futs:
            f.result()
    fio.write_bytes(done, f"{n} files {nbytes} bytes {time.time() - t:.0f}s\n".encode())
    return kind, group, "ok", n, time.time() - t


def cmd_all(args):
    import multiprocessing as mp
    jobs = []
    for kind in args.kinds:
        groups = [g[: -len(".tar.gz")] for g in list_groups(kind)]
        if args.groups:
            groups = [g for g in groups if g in set(args.groups)]
        jobs += [(kind, g, args.via_local, args.threads) for g in groups]
    print(f"[all] {len(jobs)} tar files ({', '.join(args.kinds)}) -> {DST}, procs={args.procs}", flush=True)
    t0, n_err = time.time(), 0
    with mp.get_context("fork").Pool(args.procs) as pool:
        for i, job in enumerate(pool.imap_unordered(_safe_extract, jobs), 1):
            kind, group, status, n, dt = job
            n_err += status.startswith("error")
            print(f"  [{i}/{len(jobs)}] {kind}/{group}: {status} {n} files {dt / 60:.1f} min "
                  f"(elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
    print(f"[all] finished, {n_err} errors（重跑同一条命令只补做没完成的 tar）", flush=True)
    return 1 if n_err else 0


def _safe_extract(job):
    try:
        return _extract_group(job)
    except Exception as e:
        return job[0], job[1], f"error: {e!r}"[:300], 0, 0.0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=3, help="解几个 clip")
    s.add_argument("--group", default="group_0001")
    s.add_argument("--scan-limit", type=int, default=0, help="annotations / videos 里最多扫多少个成员去找同一批 clip（0 = 扫完整个 tar）")
    s.add_argument("--via-local", default=None, help="本地目录：先把 tar.gz 拷到这里再解（流式读取不可用时）")
    i = sub.add_parser("inspect")
    i.add_argument("--clips", type=int, default=1, help="每个类别看几个 clip")
    i.add_argument("--max-files", type=int, default=40)
    a = sub.add_parser("all")
    a.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    a.add_argument("--groups", nargs="*", default=None)
    a.add_argument("--procs", type=int, default=8, help="同时解几个 tar")
    a.add_argument("--threads", type=int, default=16, help="每个 tar 并发写 OBS 的文件数")
    a.add_argument("--via-local", default=None)
    args = ap.parse_args()
    return {"sample": cmd_sample, "inspect": cmd_inspect, "all": cmd_all}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
