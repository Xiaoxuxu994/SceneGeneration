#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialVID-HQ 预处理：直接从 OBS 上的 tar.gz（annotations / depths / videos）生成按帧的训练数据，不经过中间解压。

每个 group 依次流式读三个 tar：
  1) annotations：每个 clip 的 poses.npy、intrinsics.npy、dyn_masks.npz 留在内存（每个 group 约 1.6GB 压缩）；
  2) depths：每个 clip 一个 zip（逐帧 EXR，通道 Z，float16 逆深度）-> 深度 = 1/Z，
     Z <= 0（无效）和 Z <= --sky-inv（天空，官方截在 200m）置 0，动态区域（dyn_masks=1）置 0；
  3) videos：只解有深度的 clip，按官方规则取第 k*int(fps/5) 帧，缩到深度的分辨率（584x328），存 jpg。
GT 约定（官方 README / utils）：poses 每行 [tx,ty,tz,qx,qy,qz,qw]，按官方 quat_to_mat.py 构造的是 w2c（OpenCV，米）；
intrinsics 每行 [fx,fy,cx,cy] 归一化（fx,cx 乘宽，fy,cy 乘高）。这里原样存 w2c，loader 里再求逆（pose_is_w2c）。

输出（--out，本地或 obs://）：
    <out>/<group>/<clip>/frame_<k:05d>.jpg    RGB，深度的分辨率
    <out>/<group>/<clip>/depth_<k:05d>.npy    float16 米制深度，0 = 无效（负值、天空、动态区域）
    <out>/<group>/<clip>/cams.pkl             {"<k:05d>": (K 3x3 像素, w2c 4x4)}
    <out>/<group>/<clip>/meta.json            fps、取帧间隔、帧数、分辨率、有效 / 动态像素比例
    <out>/<group>/_index.json                 {clip: 帧数}，loader 建索引只读它
    <out>/<group>/_DONE                       group 写完的标记；重跑跳过
    断点续跑：group 有 _DONE 整组跳过；否则逐 clip 检查 meta.json（最后同步写）与 jpg 数，已完成的 clip 不再解码

--pack（配合 --no-depth 使用）：每个 clip 输出 <clip>/clip.tar（frame_<k:05d>.jpg + cams.pkl + 原始标注，不压缩）+ meta.json。
--no-depth（不读 depths/，覆盖全部 74 个 group）：只出 RGB 帧 + 相机，供校验 / 重新打伪标签用；
    帧缩到长边 --max-side（默认 1280 即原分辨率），K 按帧的实际尺寸换算，
    同时原样拷贝每个 clip 的 poses.npy / intrinsics.npy / dyn_masks.npz / caption.json / instructions.json。

用法（仓库根目录）：
    python tools/preprocess_spatialvid.py --groups group_0001 --limit-clips 5      # 先试几个 clip
    python tools/preprocess_spatialvid.py --procs 6                                # 全部有深度的 group
    python tools/preprocess_spatialvid.py --no-depth --procs 8                     # 全部 group，只要 RGB + pose
"""
import argparse
import io
import json
import os
import os.path as osp
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import spatialvid_extract as SX  # noqa: E402  (tar 流式读取、clip id；会把工程根加入 sys.path)
from datasets.base import fileio as fio  # noqa: E402

ARGS = None
DEFAULT_OUT = "obs://bucket-471/z00533370/dataset/SpatialVID-HQ_frames"
DEFAULT_OUT_NODEPTH = "obs://bucket-471/z00533370/dataset/SpatialVID-HQ_rgbpose"
ANN_FILES = ("poses.npy", "intrinsics.npy", "dyn_masks.npz", "caption.json", "instructions.json")


def quat_to_R(q):
    """q = (qx, qy, qz, qw) -> 3x3（右手系，与官方 quaternion_to_matrix 的 (i, j, k, r) 顺序一致）。"""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def read_exr_inv_depth(data):
    import OpenEXR
    exr = OpenEXR.InputFile(io.BytesIO(data))
    dw = exr.header()["dataWindow"]
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
    return np.frombuffer(exr.channels(["Z"])[0], dtype=np.float16).reshape(h, w).astype(np.float32)


def expand_dyn_masks(data):
    """官方 expand_npz.py：每帧一个 CSR（f_i_data / f_i_indices / f_i_indptr）+ 共用 shape。"""
    from scipy.sparse import csr_matrix
    z = np.load(io.BytesIO(data))
    shape = tuple(int(v) for v in z["shape"])
    n = (len(z.files) - 1) // 3
    return [csr_matrix((z[f"f_{i}_data"], z[f"f_{i}_indices"], z[f"f_{i}_indptr"]), shape=shape) for i in range(n)]


def make_cam(pose_row, intr_row, W, H):
    """官方 poses 一行 [tx,ty,tz,qx,qy,qz,qw] -> w2c 4x4；归一化内参一行 [fx,fy,cx,cy] -> 像素 K（宽 W、高 H）。"""
    fx, fy, cx, cy = (float(v) for v in intr_row)
    K = np.array([[fx * W, 0, cx * W], [0, fy * H, cy * H], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = quat_to_R(pose_row[3:7])
    w2c[:3, 3] = pose_row[:3]
    return K, w2c


def n_dyn_masks(data):
    return (len(np.load(io.BytesIO(data)).files) - 1) // 3


def process_depth_clip(clip, data, ann, out_dir, write):
    """返回 clip 的元信息（K / w2c / 分辨率 / 统计），深度帧已提交写出。"""
    import cv2
    poses = np.load(io.BytesIO(ann["poses.npy"]))
    intr = np.load(io.BytesIO(ann["intrinsics.npy"]))
    masks = expand_dyn_masks(ann["dyn_masks.npz"]) if "dyn_masks.npz" in ann else None
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = sorted(n for n in zf.namelist() if n.lower().endswith(".exr"))
    n = min(len(names), len(poses), len(intr), len(masks) if masks is not None else 10 ** 9)
    cams, W, H, valid, dyn = {}, None, None, [], []
    for k in range(n):
        inv = read_exr_inv_depth(zf.read(names[k]))
        H, W = inv.shape
        ok = np.isfinite(inv) & (inv > ARGS.sky_inv)
        depth = np.where(ok, 1.0 / np.where(ok, inv, 1.0), 0.0).astype(np.float32)
        if masks is not None:
            m = cv2.resize(masks[k].toarray().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            dyn.append(float(m.mean()))
            depth[m] = 0
        valid.append(float((depth > 0).mean()))
        key = f"{k:05d}"
        cams[key] = make_cam(poses[k], intr[k], W, H)
        write(osp.join(out_dir, f"depth_{key}.npy"), fio.np_save_bytes(depth.astype(np.float16)))
    return dict(n=n, W=W, H=H, cams=cams, n_depth=len(names), n_pose=len(poses),
                valid=float(np.mean(valid)) if valid else 0.0, dyn=float(np.mean(dyn)) if dyn else 0.0)


def clip_done(out_dir):
    """meta.json 在该 clip 的其余文件全部写完后才写（同步），且帧数与目录里的 jpg 数一致 -> 已完成，返回帧数。"""
    try:
        if not fio.exists(osp.join(out_dir, "meta.json")):
            return None
        n = int(json.loads(fio.read_bytes(osp.join(out_dir, "meta.json")))["n_frames"])
        if ARGS.pack:                                # clip.tar 同步写完才写 meta.json
            return n if fio.exists(osp.join(out_dir, "clip.tar")) else None
        names = fio.listdir(out_dir)
        need = {"cams.pkl"} | (set(ANN_FILES) - {"caption.json", "instructions.json"} if ARGS.no_depth else set())
        if need <= set(names) and sum(x.startswith("frame_") for x in names) == n:
            return n
    except Exception:
        pass
    return None


def process_video_clip(clip, data, info, out_dir, write):
    """返回 (帧数, 是否沿用了已完成的输出)。"""
    import cv2
    cv2.setNumThreads(1)
    if not ARGS.overwrite:
        n = clip_done(out_dir)
        if n is not None:
            return n, True
    mine, packed = [], {}

    def write(path, payload, _write=write):
        if ARGS.pack:                                # --pack：先收进内存，最后打成一个 clip.tar 上传
            packed[osp.basename(path)] = payload
        else:
            mine.append(_write(path, payload))

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        tf.write(data)
        tf.flush()
        cap = cv2.VideoCapture(tf.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(int(fps / 5), 1)
        n_img, i = 0, 0
        while n_img < info["n"]:
            if not cap.grab():
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if info["W"] is None:              # --no-depth：按长边 --max-side 定尺寸，再算 K
                    h0, w0 = frame.shape[:2]
                    sc = min(1.0, ARGS.max_side / max(h0, w0))
                    info["W"], info["H"] = int(round(w0 * sc)), int(round(h0 * sc))
                    poses, intr = np.load(io.BytesIO(info["raw"]["poses.npy"])), np.load(io.BytesIO(info["raw"]["intrinsics.npy"]))
                    info["cams"] = {f"{k:05d}": make_cam(poses[k], intr[k], info["W"], info["H"]) for k in range(info["n"])}
                if (frame.shape[1], frame.shape[0]) != (info["W"], info["H"]):
                    frame = cv2.resize(frame, (info["W"], info["H"]), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, ARGS.jpeg_quality])
                write(osp.join(out_dir, f"frame_{n_img:05d}.jpg"), buf.tobytes())
                n_img += 1
            i += 1
        cap.release()
    n = min(n_img, info["n"])
    cams = {k: v for k, v in (info["cams"] or {}).items() if int(k) < n}
    for fn, raw in (info.get("raw") or {}).items():           # --no-depth：原始标注原样拷贝
        write(osp.join(out_dir, fn), raw)
    if ARGS.pack:
        import pickle
        packed["cams.pkl"] = pickle.dumps(cams)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:      # 不压缩：jpg 已压缩，读取时可直接按成员取
            for name in sorted(packed):
                ti = tarfile.TarInfo(name)
                ti.size = len(packed[name])
                tar.addfile(ti, io.BytesIO(packed[name]))
        fio.write_bytes(osp.join(out_dir, "clip.tar"), buf.getvalue())
    else:
        fio.save_pickle(cams, osp.join(out_dir, "cams.pkl"))
    for f in mine:                                   # 这个 clip 的文件全部写完，再同步写 meta.json（完成标记）
        f.result()
    meta = dict(clip=clip, fps=fps, step=step, n_frames=n, n_video_frames_read=i, n_depth=info["n_depth"],
                n_pose=info["n_pose"], W=info["W"], H=info["H"], valid_ratio=info["valid"], dyn_ratio=info["dyn"])
    fio.write_bytes(osp.join(out_dir, "meta.json"), json.dumps(meta).encode())
    return n, False


def process_group(group):
    t0 = time.time()
    out_g = osp.join(ARGS.out, group)
    try:
        if not ARGS.overwrite and fio.exists(osp.join(out_g, "_DONE")):
            return group, "skip", 0, 0.0
        pool = ThreadPoolExecutor(ARGS.threads)
        futs, lock = [], threading.Lock()

        def write(path, data):
            """异步写 OBS；待写的文件数超过上限时等最早的写完（深度帧与视频解码线程共用）。"""
            with lock:
                fut = pool.submit(fio.write_bytes, path, data)
                futs.append(fut)
                wait = futs[: -2 * ARGS.threads] if len(futs) > 8 * ARGS.threads else []
                del futs[: len(wait)]
            for f in wait:
                f.result()
            return fut

        # 1) annotations -> 内存
        ann = {}
        for name, data in SX.iter_members(f"{SX.SRC}/annotations/{group}.tar.gz", ARGS.via_local):
            fn = name.rsplit("/", 1)[-1]
            keep = ANN_FILES if ARGS.no_depth else ("poses.npy", "intrinsics.npy", "dyn_masks.npz")
            if fn in keep:
                ann.setdefault(SX.clip_key(name), {})[fn] = data
        print(f"[{group}] annotations: {len(ann)} clips ({time.time() - t0:.0f}s)", flush=True)

        # 2) depths -> 深度帧 + 相机（--no-depth：直接用 annotations 的 clip，相机在解视频时按帧尺寸算）
        infos, n_no_ann, n_bad = {}, 0, 0
        if ARGS.no_depth:
            for clip, a in ann.items():
                if not {"poses.npy", "intrinsics.npy"} <= set(a):
                    n_no_ann += 1
                    continue
                n_pose = len(np.load(io.BytesIO(a["poses.npy"])))
                n = min(n_pose, len(np.load(io.BytesIO(a["intrinsics.npy"]))),
                        n_dyn_masks(a["dyn_masks.npz"]) if "dyn_masks.npz" in a else 10 ** 9)
                if n >= ARGS.min_frames:
                    infos[clip] = dict(n=n, W=None, H=None, cams=None, n_depth=0, n_pose=n_pose, valid=None, dyn=None, raw=a)
                if ARGS.limit_clips and len(infos) >= ARGS.limit_clips:
                    break
        depth_tar = [] if ARGS.no_depth else SX.iter_members(f"{SX.SRC}/depths/{group}.tar.gz", ARGS.via_local)
        for name, data in depth_tar:
            clip = SX.clip_key(name)
            if clip not in ann or not {"poses.npy", "intrinsics.npy"} <= set(ann[clip]):
                n_no_ann += 1
                continue
            try:
                info = process_depth_clip(clip, data, ann[clip], osp.join(out_g, clip), write)
            except Exception as e:
                n_bad += 1
                print(f"[{group}] depth {clip}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            if info["n"] >= ARGS.min_frames:
                infos[clip] = info
            if ARGS.limit_clips and len(infos) >= ARGS.limit_clips:
                break
        del ann
        print(f"[{group}] {'clips with pose' if ARGS.no_depth else 'depths'}: {len(infos)} clips (no annotation {n_no_ann}, failed {n_bad}) ({time.time() - t0:.0f}s)",
              flush=True)

        # 3) videos -> RGB 帧（只解有深度的 clip）
        index, vpool = {}, ThreadPoolExecutor(ARGS.video_workers)
        vfuts = {}
        for name, data in SX.iter_members(f"{SX.SRC}/videos/{group}.tar.gz", ARGS.via_local):
            clip = SX.clip_key(name)
            if clip in infos and clip not in vfuts:
                vfuts[clip] = vpool.submit(process_video_clip, clip, data, infos[clip], osp.join(out_g, clip), write)
                if len(vfuts) % 500 == 0:
                    print(f"[{group}] videos: {len(vfuts)}/{len(infos)} clips decoding ({time.time() - t0:.0f}s)", flush=True)
                while sum(not f.done() for f in vfuts.values()) > 4 * ARGS.video_workers:
                    time.sleep(0.05)
                if len(vfuts) == len(infos):
                    break
        n_reused = 0
        for clip, f in vfuts.items():
            try:
                n, reused = f.result()
                n_reused += reused
                if n >= ARGS.min_frames:
                    index[clip] = n
            except Exception as e:
                print(f"[{group}] video {clip}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        vpool.shutdown()
        for f in futs:
            f.result()
        pool.shutdown()
        missing = sorted(set(infos) - set(vfuts))
        fio.write_bytes(osp.join(out_g, "_index.json"), json.dumps(index).encode())
        if not ARGS.limit_clips:
            fio.write_bytes(osp.join(out_g, "_DONE"),
                            f"{len(index)} clips, {sum(index.values())} frames, no video {len(missing)}\n".encode())
        print(f"[{group}] videos: {len(index)} clips written ({n_reused} reused from an earlier run), {sum(index.values())} frames, without video "
              f"{len(missing)} ({time.time() - t0:.0f}s)", flush=True)
        return group, "ok", len(index), time.time() - t0
    except Exception as e:
        return group, f"error {type(e).__name__}: {str(e)[:200]}", 0, time.time() - t0


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"默认 {DEFAULT_OUT}；--no-depth 时默认 {DEFAULT_OUT_NODEPTH}")
    ap.add_argument("--groups", nargs="*", default=None, help="默认：depths/ 下现有的全部 group（--no-depth 时为 videos/ 下全部）")
    ap.add_argument("--no-depth", action="store_true", help="不读深度，只出 RGB 帧 + 相机 + 原始标注")
    ap.add_argument("--pack", action="store_true",
                    help="每个 clip 打成一个 clip.tar（帧 jpg + cams.pkl + 原始标注）再上传；OBS 小文件写入单节点约 300 个/s 封顶，"
                         "逐帧写是瓶颈（2026-11-11 实测），打包后每个 clip 只有 clip.tar + meta.json 两个对象")
    ap.add_argument("--max-side", type=int, default=1280, help="--no-depth 时帧的长边（默认 1280 = 原分辨率）")
    ap.add_argument("--limit-clips", type=int, default=0, help=">0 时每个 group 只做前 N 个有深度的 clip（试跑，不写 _DONE）")
    ap.add_argument("--procs", type=int, default=6, help="同时处理几个 group")
    ap.add_argument("--threads", type=int, default=16, help="每个 group 并发写 OBS 的文件数")
    ap.add_argument("--video-workers", type=int, default=4, help="每个 group 并发解码的视频数")
    ap.add_argument("--decode-threads", type=int, default=1,
                    help="每个视频解码用的 ffmpeg 线程数（OpenCV 默认按核数开，几十个视频并发时严重超订；并行度靠 procs×video-workers）")
    ap.add_argument("--sky-inv", type=float, default=0.0052, help="逆深度 <= 此值（约 192m 以外，官方把天空截在 200m）置为无效")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--via-local", default=None, help="本地目录：先把 tar.gz 拷到这里再解")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n：多节点时本节点处理第 i 份（group 列表按 i::n 切分）")
    ARGS = ap.parse_args()
    # OpenCV 打开视频时读这个环境变量；fork 出的进程继承
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"threads;{ARGS.decode_threads}"

    if ARGS.no_depth and ARGS.out == DEFAULT_OUT:
        ARGS.out = DEFAULT_OUT_NODEPTH
    groups = ARGS.groups or [g[: -len(".tar.gz")] for g in SX.list_groups("videos" if ARGS.no_depth else "depths")]
    si, sn = (int(x) for x in ARGS.shard.split("/"))
    groups = groups[si::sn]
    print(f"[SpatialVID] {len(groups)} groups -> {ARGS.out}  procs={ARGS.procs} threads={ARGS.threads} "
          f"video_workers={ARGS.video_workers} limit_clips={ARGS.limit_clips} shard={ARGS.shard} no_depth={ARGS.no_depth} "
          f"max_side={ARGS.max_side} jpeg_quality={ARGS.jpeg_quality} decode_threads={ARGS.decode_threads}", flush=True)
    print(f"[SpatialVID] groups: {' '.join(groups)}", flush=True)
    import multiprocessing as mp
    t0, n_err = time.time(), 0
    with mp.get_context("fork").Pool(ARGS.procs) as pool:
        for i, (group, status, n, dt) in enumerate(pool.imap_unordered(process_group, groups), 1):
            n_err += status.startswith("error")
            print(f"[SpatialVID] [{i}/{len(groups)}] {group}: {status} {n} clips {dt / 60:.1f} min "
                  f"(elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
    print(f"[SpatialVID] finished, {n_err} groups with errors（重跑同一条命令只补做没完成的 group）", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
