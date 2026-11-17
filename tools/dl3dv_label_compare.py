#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL3DV 新旧标签对比：同一场景、同一帧名，把 Pi3 训练原来读的标签（DL3DV_WAI）和 GLR-v3 新打标结果放在一起看。

旧标签 = datasets/dl3dv_dataset.py::DL3DVDataset 实际读的东西（裁剪/缩放之前那一步）：
    DL3DV_WAI/<子集>-<hash>/scene_meta.json   位姿 transform_matrix（c2w，OpenCV 轴）、场景级内参 fl_x/fl_y/cx/cy
    .../<frame>.mvsanywhere_depth (+ _confidence) 深度 + 置信度
    DL3DV/<子集>/<hash>/images_8/<帧名>.png     RGB（原始带畸变的 480P）
  处理与 _load_one_view 相同：深度 resize 到 RGB 尺寸（lanczos/nearest）、nan/<=0 置 0、98 分位截断一次（pct_twice），
  训练掩码 = conf > conf_thres & depth > depth_thres。不做 _crop_resize_if_necessary（随机裁剪会让两边对不上）。

新标签 = GLR-v3（Agentic-Anno，ROMA 打标）每场景结果 tar：<out-root>/<run>/results/<子集>/<hash>.tar
    <hash>/project/adaptive/project/sparse/0/{cameras,images}.bin   去畸变 PINHOLE 相机 + 位姿（known-poses：位姿来自同一份 transforms.json）
    .../depth/*.npz {depth, valid_mask}   RoMa 真实观测三角化后投影的深度（OpenCV 相机 Z，SfM 单位；稀疏，不是 PDA 补全）
    .../images/<帧名>.png                  去畸变后的 RGB
    .../sparse/0/points3D.ply              最终点云

三步（各自只读，可在不同机器上跑，中间产物是每场景一个小 npz）：
    # 1) 能挂 east4-2 的机器：抽旧标签，同时决定抽哪几帧（写 OUT/frames.json）
    python tools/dl3dv_label_compare.py old --scenes list.txt --out OUT --n-frames 8
    # 2) 能读 OBS 结果 tar 的机器（ROMA 镜像里有 moxing）：按同一份 frames.json 抽新标签
    python tools/dl3dv_label_compare.py new --scenes list.txt --out OUT \
        --results obs://bucket-471/z00533370/dataset/DL3DV-test/glr/dl3dv_s256/results [--frames-from OUT/frames.json]
  固定场景/帧的一键版本见 tools/dl3dv_compare/run.sh（old / new 各一条命令，结束后打成 tar.gz 便于传出）。
    # 3) 任意机器（Mac 也行，只要 numpy/opencv/matplotlib）：出报告
    python tools/dl3dv_label_compare.py report --out OUT        # -> OUT/report/index.html
  一台机器两边都能读时：  ... all --scenes list.txt --out OUT --results <...>
    # 4) 批量扫描（旧标签疑似失败、新标签可靠的场景）：old 步加 --neighbor-offset 4 再跑 scan
    python tools/dl3dv_label_compare.py scan --out OUT --top 24   # -> OUT/scan/{scan.csv, scan_frames.csv, panels/, 3d/}
  打分规则见 SCAN_RULES；分批跑见 tools/dl3dv_compare/run.sh scan-new / scan-old / scan。

场景清单：每行 `1K/<64位hash>`（Agentic-Anno tools/dl3dv/pilot/scenes_*.txt 的格式），也接受 `1K-<hash>`；# 开头为注释。
也可以 `--ledger-dir <E/roma/runs/<run>>` 直接用打标台账里 class=complete 的场景（只在 new/all 有用）。

报告里每个场景回答四个问题：
  1. 位姿是不是同一套：旧 c2w 与新位姿按相机中心做 Sim3 对齐后的旋转/中心残差（known-poses 应接近 0，否则是约定没对上）
  2. 旧深度的尺度和位姿是否一致：把新深度（与位姿同一 SfM 单位）投到旧相机，看 新/旧 中位比；
     比值 ≈ Sim3 尺度说明一致，帧间比值的离散度说明旧深度逐帧尺度不稳
  3. 旧深度准不准：按场景统一尺度后的 AbsRel、δ<1.05 / 1.10 / 1.25（只在新深度有观测的像素上）
  4. 旧 RGB 与旧深度/内参是否对齐：分别用「训练假设的针孔」和「带畸变」两种方式把新点投进旧图，哪种和旧深度更吻合
     就说明旧深度在哪个图像网格上；配合 RGB 叠深度边缘的图直接看错位
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import os.path as osp
import posixpath
import shutil
import struct
import sys
import tarfile
import tempfile
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np  # noqa: E402

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
WAI_ROOT = '/efs-mi-east4-2/datasets/DL3DV_WAI'
RGB_ROOT = '/efs-mi-east4-2/datasets/DL3DV'
PROJECT_CANDIDATES = ('project/adaptive/project', 'project/project', 'project')
CAMERA_PARAMS = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4), 3: ('RADIAL', 5),
                 4: ('OPENCV', 8), 5: ('OPENCV_FISHEYE', 8), 6: ('FULL_OPENCV', 12), 7: ('FOV', 5),
                 8: ('SIMPLE_RADIAL_FISHEYE', 4), 9: ('RADIAL_FISHEYE', 5), 10: ('THIN_PRISM_FISHEYE', 12),
                 11: ('RAD_TAN_THIN_PRISM_FISHEYE', 16)}


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ============================================================================ 文件读写（本地 / obs://）
class IO:
    """训练仓库的 datasets.base.fileio / read_any_file / threshold_depth_map；导入失败（如 Mac 上没有 torch）时用等价的本地实现。"""

    def __init__(self, require_repo=False):
        self.repo = False
        try:
            if REPO not in sys.path:
                sys.path.insert(0, REPO)
            from datasets.base import fileio as fio
            from datasets.base.utils import read_any_file
            from datasets.sample_utils import common
            self.fio, self._read_any, self._common = fio, read_any_file, common
            self.repo = True
        except Exception as exc:  # noqa: BLE001
            if require_repo:
                raise SystemExit(f'导入训练仓库的读文件函数失败（{type(exc).__name__}: {exc}）；old 步骤要与训练一致，'
                                 f'请在训练环境里、仓库根目录下运行，或加 --allow-local-io')
            log(f'注意：没导入训练仓库的 fileio（{type(exc).__name__}: {exc}），只能读本地文件')

    def exists(self, p):
        return self.fio.exists(p) if self.repo else osp.exists(p)

    def read_bytes(self, p):
        if self.repo:
            return self.fio.read_bytes(p)
        with open(p, 'rb') as fh:
            return fh.read()

    def read_json(self, p):
        return json.loads(self.read_bytes(p))

    def listdir(self, p):
        return self.fio.listdir(p) if self.repo else os.listdir(p)

    def read_any(self, p, return_dtype=None):
        if self.repo:
            return self._read_any(p, return_dtype=return_dtype)
        import cv2
        ext = osp.splitext(p)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg', '.ppm'):
            from PIL import Image
            with Image.open(p) as im:
                return np.array(im, return_dtype)
        if ext in ('.exr', '.tif', '.tiff'):
            return np.asarray(cv2.imread(p, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR), return_dtype)
        if ext in ('.npy', '.npz'):
            return np.load(p)
        raise NotImplementedError(p)

    def threshold(self, d, max_percentile):
        if self.repo:
            return self._common.threshold_depth_map(d, min_percentile=0, max_percentile=max_percentile)
        d = d.astype(float, copy=True)          # 默认开关（valid_only=False, disabled=False）下的同一逻辑
        thr = np.nanpercentile(d, max_percentile) if max_percentile > 0 else None
        if thr is not None and thr > 0:
            d[d > thr] = 0.0
        return d

    def copy_to_local(self, src, dst):
        if src.startswith('obs://'):
            if self.repo:
                self.fio._mox().file.copy(self.fio.resolve(src), dst)
                return
            # new 步骤只需要下载 tar：训练仓库导入失败（如 ROMA 镜像没有 hydra）时直接用 moxing
            if not hasattr(self, '_mox'):
                try:
                    import moxing as mox
                    mox.file.set_auth(path_style=True, is_secure=False, retry=10, retry_wait=1)
                    self._mox = mox
                    log('训练仓库没导入，改用 moxing 直接读 obs://')
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(f'读 obs:// 需要 moxing（{type(exc).__name__}: {exc}）')
            self._mox.file.copy(src, dst)
        else:
            shutil.copyfile(src, dst)


def join(a, *b):
    return posixpath.join(a, *b) if a.startswith('obs://') else osp.join(a, *b)


# ============================================================================ 场景 / 帧
def norm_sid(s):
    s = s.strip().rstrip('/')
    if not s or s.startswith('#'):
        return None
    s = s.split()[0]
    if '/' in s:
        sub, h = s.split('/')[-2:]
    elif '-' in s:
        sub, h = s.split('-', 1)
    else:
        raise SystemExit(f'场景名要带子集：1K/<hash> 或 1K-<hash>，收到 {s!r}')
    return f'{sub}/{h}'


def read_scenes(args):
    sids = []
    for f in args.scenes or []:
        with open(f) as fh:
            sids += [x for x in map(norm_sid, fh) if x]
    if getattr(args, 'ledger_dir', None):
        rows = load_ledger(args.ledger_dir)
        done = sorted(k for k, r in rows.items() if r.get('class') == 'complete')
        log(f'台账 {args.ledger_dir}: {len(rows)} 行，complete {len(done)}')
        sids = [s for s in sids if s in rows] if sids else done
    for s in getattr(args, 'sids', None) or []:
        sids.append(norm_sid(s))
    seen, out = set(), []
    for s in sids:
        if s not in seen:
            seen.add(s)
            out.append(s)
    if args.limit:
        out = spread(out, args.limit)
    if not out:
        raise SystemExit('没有场景：给 --scenes / --sids / --ledger-dir')
    sh = getattr(args, 'shard', '') or ''
    if sh:
        i, n = (int(x) for x in sh.split('/'))
        out = out[i::n]
        log(f'分片 {i}/{n}：{len(out)} 个场景')
    return out


def load_ledger(d):
    rows = {}
    for f in sorted(os.listdir(d)):
        if f.startswith('ledger_node') and f.endswith('.jsonl'):
            with open(osp.join(d, f)) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        rows[r['sid']] = r         # 同一场景多行时最后一行为准（与 view_scenes 相同）
                    except (ValueError, KeyError, TypeError):
                        continue
    return rows


def spread(items, n):
    if n <= 0 or len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def stem(name):
    return osp.splitext(osp.basename(name))[0]


def safe(sid):
    sub, h = sid.split('/')
    return f'{sub}_{h[:12]}'


def load_frames_json(out):
    p = osp.join(out, 'frames.json')
    if osp.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {}


def save_frames_json(out, frames):
    import fcntl
    p = osp.join(out, 'frames.json')
    with open(p + '.lock', 'w') as lk:    # 多个分片进程同时写时加锁
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = load_frames_json(out)
        for k, v in frames.items():
            cur.setdefault(k, v)              # 已有的选择不覆盖：两边必须抽同一组帧
        tmp = p + f'.tmp{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump(cur, fh, indent=1, sort_keys=True)
        os.replace(tmp, p)


def pick_frames(available, sid, args, frames_json):
    """available: 帧名 stem 列表（已排序）。优先用 frames.json 里已有的选择，其次 --frames，再次均匀抽 n 帧。"""
    if sid in frames_json:
        want = frames_json[sid]
    elif args.frames:
        want = [stem(x) for x in args.frames.split(',')]
    else:
        want = spread(sorted(available), args.n_frames)
    have = set(available)
    return [w for w in want if w in have], [w for w in want if w not in have]


# ============================================================================ 1) 旧标签（DL3DV_WAI，与 DL3DVDataset 相同）
def run_old(args):
    io_ = IO(require_repo=not args.allow_local_io)
    import cv2
    sids = read_scenes(args)
    os.makedirs(osp.join(args.out, 'old'), exist_ok=True)
    frames_json = load_frames_json(args.out)
    if args.frames_from:
        with open(args.frames_from) as fh:
            frames_json = {**json.load(fh), **frames_json}
    picked, status = {}, []
    for i, sid in enumerate(sids):
        sub, h = sid.split('/')
        dst = osp.join(args.out, 'old', safe(sid) + '.npz')
        if osp.exists(dst) and not args.force:
            status.append(dict(sid=sid, status='skip_existing'))
            continue
        wai = join(args.wai_root, f'{sub}-{h}')
        t0 = time.time()
        try:
            raw_meta = io_.read_bytes(join(wai, 'scene_meta.json'))
            with open(osp.join(args.out, 'old', safe(sid) + '_scene_meta.json'), 'wb') as fh:
                fh.write(raw_meta)                # 原样留一份：相机字段/畸变/帧字段都以它为准
            meta = json.loads(raw_meta)
            dmeta = None
            if io_.exists(join(wai, 'scene_meta_distorted.json')):
                dmeta = io_.read_json(join(wai, 'scene_meta_distorted.json'))
            if meta.get('scene_name') not in (None, f'{sub}-{h}'):
                raise RuntimeError(f"scene_name={meta.get('scene_name')} 与目录不符（DL3DVDataset 会跳过这个场景）")
            K = np.eye(3)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = (float(meta[k]) for k in ('fl_x', 'fl_y', 'cx', 'cy'))
            by = {}
            for fr in meta['frames']:
                by[stem(fr['file_path'])] = fr
            names, miss = pick_frames(sorted(by), sid, args, frames_json)
            picked[sid] = names + miss
            rgbs, depths, raws, confs, masks, c2ws = [], [], [], [], [], []
            nb_names, nb_depths, nb_confs, nb_c2ws = [], [], [], []
            order = sorted(by)
            for n in names:
                fr = by[n]
                rgb_path = join(args.rgb_root, sub, h, args.rgb_subdir, n + args.rgb_ext)
                rgb = io_.read_any(rgb_path)
                if rgb.ndim == 2:
                    rgb = np.repeat(rgb[..., None], 3, -1)
                rgb = rgb[..., :3].astype(np.uint8)
                H, W = rgb.shape[:2]
                if args.neighbor_offset:
                    # 相邻帧（只要深度/置信度/位姿）：用来查旧深度自身的多视图一致性，不依赖新标签
                    k = order.index(n)
                    kn = k + args.neighbor_offset if k + args.neighbor_offset < len(order) else k - args.neighbor_offset
                    if 0 <= kn < len(order) and kn != k:
                        frn = by[order[kn]]
                        dn = np.asarray(io_.read_any(join(wai, frn['mvsanywhere_depth']), return_dtype=np.float32), np.float32)
                        dn = dn[..., 0] if dn.ndim == 3 else dn
                        dn = cv2.resize(dn, (W, H), interpolation=cv2.INTER_NEAREST if args.depth_interp == 'nearest' else cv2.INTER_LANCZOS4)
                        dn = np.nan_to_num(dn, nan=0.0, posinf=0.0, neginf=0.0)
                        dn[dn <= 0] = 0.0
                        if args.pct_twice:
                            dn = io_.threshold(dn, 98)
                        cn = np.asarray(io_.read_any(join(wai, frn['mvsanywhere_depth_confidence']), return_dtype=np.float32), np.float32)
                        cn = cn[..., 0] if cn.ndim == 3 else cn
                        if cn.shape[:2] != (H, W):
                            cn = cv2.resize(cn, (W, H), interpolation=cv2.INTER_NEAREST)
                        nb_names.append(order[kn]); nb_depths.append(dn.astype(np.float16)); nb_confs.append(cn.astype(np.float16))
                        nb_c2ws.append(np.array(frn['transform_matrix'], np.float64))
                    else:
                        nb_names.append(''); nb_depths.append(np.zeros((H, W), np.float16))
                        nb_confs.append(np.zeros((H, W), np.float16)); nb_c2ws.append(np.eye(4))
                d = io_.read_any(join(wai, fr['mvsanywhere_depth']), return_dtype=np.float32)
                d = np.asarray(d, np.float32)
                if d.ndim == 3:
                    d = d[..., 0]
                raw_hw = d.shape
                interp = cv2.INTER_NEAREST if args.depth_interp == 'nearest' else cv2.INTER_LANCZOS4
                d = cv2.resize(d, (W, H), interpolation=interp)
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                d[d <= 0] = 0.0
                raw = d.copy()
                c = np.asarray(io_.read_any(join(wai, fr['mvsanywhere_depth_confidence']), return_dtype=np.float32), np.float32)
                if c.ndim == 3:
                    c = c[..., 0]
                if c.shape[:2] != (H, W):
                    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)
                if args.pct_twice:
                    d = io_.threshold(d, 98)
                m = (c > args.conf_thres) & (d > args.depth_thres)
                c2w = np.array(fr['transform_matrix'], np.float64)
                if args.apply_trans:
                    t1 = np.array(meta['_applied_transform'])
                    t2 = np.array(meta['_applied_transforms']['opengl2opencv'])
                    c2w = t2 @ t1 @ c2w
                rgbs.append(rgb), depths.append(d.astype(np.float32)), raws.append(raw.astype(np.float16))
                confs.append(c.astype(np.float16)), masks.append(m), c2ws.append(c2w)
            all_names = sorted(by)
            all_c2w = np.stack([np.array(by[n]['transform_matrix'], np.float64) for n in all_names])
            cam = {k: v for k, v in meta.items() if k != 'frames' and not isinstance(v, (list, dict))}
            dcam = {k: v for k, v in (dmeta or {}).items() if k != 'frames' and not isinstance(v, (list, dict))}
            info = dict(sid=sid, wai=wai, rgb_root=args.rgb_root, rgb_subdir=args.rgb_subdir, camera=cam,
                        camera_distorted=dcam or None, depth_raw_hw=list(raw_hw) if names else None,
                        rgb_hw=list(rgbs[0].shape[:2]) if rgbs else None, missing=miss,
                        loader=dict(conf_thres=args.conf_thres, depth_thres=args.depth_thres, depth_interp=args.depth_interp,
                                    pct_twice=args.pct_twice, apply_trans=args.apply_trans, repo_io=io_.repo),
                        n_frames_meta=len(all_names))
            if not names:
                raise RuntimeError(f'选中的帧在 scene_meta 里都不存在：{miss[:4]}')
            np.savez_compressed(dst + '.tmp.npz', names=np.array(names), rgb=np.stack(rgbs), depth=np.stack(depths),
                                depth_raw=np.stack(raws), conf=np.stack(confs), mask=np.stack(masks), K=K,
                                c2w=np.stack(c2ws), all_names=np.array(all_names), all_c2w=all_c2w,
                                info=np.array(json.dumps(info, default=str)),
                                **(dict(nb_names=np.array(nb_names), nb_depth=np.stack(nb_depths), nb_conf=np.stack(nb_confs),
                                        nb_c2w=np.stack(nb_c2ws)) if nb_names else {}))
            os.replace(dst + '.tmp.npz', dst)
            status.append(dict(sid=sid, status='ok', frames=len(names), missing=len(miss), sec=round(time.time() - t0, 1)))
            log(f'old {i + 1}/{len(sids)} {sid} 帧 {len(names)}（缺 {len(miss)}）{time.time() - t0:.1f}s')
        except Exception as exc:  # noqa: BLE001
            status.append(dict(sid=sid, status='error', error=f'{type(exc).__name__}: {exc}'[:300]))
            log(f'old {i + 1}/{len(sids)} {sid} 失败：{type(exc).__name__}: {exc}')
    save_frames_json(args.out, picked)
    _write_status(args.out, 'old', status, getattr(args, 'shard', ''))


# ============================================================================ 2) 新标签（GLR-v3 结果 tar）
class TarView:
    """只读 tar 视图：按 archive 路径取成员，并在 archive 内跟随（相对）符号链接。"""

    def __init__(self, path):
        self.tf = tarfile.open(path)
        self.m = {posixpath.normpath(ti.name): ti for ti in self.tf.getmembers()}

    def member(self, name, depth=0):
        ti = self.m.get(posixpath.normpath(name))
        if ti is None:
            parts = posixpath.normpath(name).split('/')       # 父目录是符号链接的情况
            for k in range(len(parts) - 1, 0, -1):
                head = '/'.join(parts[:k])
                t = self.m.get(head)
                if t is not None and t.issym() and depth < 8:
                    tgt = posixpath.normpath(posixpath.join(posixpath.dirname(head), t.linkname))
                    return self.member('/'.join([tgt] + parts[k:]), depth + 1)
            return None
        if ti.issym() and depth < 8:
            tgt = posixpath.normpath(posixpath.join(posixpath.dirname(ti.name), ti.linkname))
            return self.member(tgt, depth + 1)
        return ti

    def read(self, name):
        ti = self.member(name)
        if ti is None or not ti.isfile():
            return None
        return self.tf.extractfile(ti).read()

    def listdir(self, d):
        d = posixpath.normpath(d)
        ti = self.m.get(d)
        if ti is not None and ti.issym():
            d = posixpath.normpath(posixpath.join(posixpath.dirname(d), ti.linkname))
        pre = d + '/'
        return sorted({n[len(pre):].split('/')[0] for n in self.m if n.startswith(pre)})


def read_cameras_bin(b):
    cams, off = {}, 8
    (n,) = struct.unpack_from('<Q', b, 0)
    for _ in range(n):
        cid, mid, w, h = struct.unpack_from('<iiQQ', b, off)
        off += 24
        name, k = CAMERA_PARAMS[mid]
        params = struct.unpack_from(f'<{k}d', b, off)
        off += 8 * k
        cams[cid] = dict(model=name, width=w, height=h, params=list(params))
    return cams


def read_images_bin(b):
    ims, off = [], 8
    (n,) = struct.unpack_from('<Q', b, 0)
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from('<I7dI', b, off)
        off += 64
        end = b.index(b'\0', off)
        name = b[off:end].decode('utf-8', 'replace')
        off = end + 1
        (npts,) = struct.unpack_from('<Q', b, off)
        off += 8 + 24 * npts
        ims.append(dict(id=iid, q=(qw, qx, qy, qz), t=(tx, ty, tz), camera_id=cid, name=name))
    return sorted(ims, key=lambda im: im['name'])


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def cam_K(cam):
    p, m = cam['params'], cam['model']
    if m in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE', 'FOV'):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:
        fx, fy, cx, cy = p[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def read_ply_xyzrgb(b, max_points, rng):
    head_end = b.index(b'end_header') + len(b'end_header')
    head_end = b.index(b'\n', head_end) + 1
    lines = b[:head_end].decode('ascii', 'replace').splitlines()
    fmt, count, props, in_vertex = None, 0, [], False
    types = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2', 'ushort': 'u2',
             'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4', 'float': 'f4', 'float32': 'f4',
             'double': 'f8', 'float64': 'f8'}
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == 'format':
            fmt = t[1]
        elif t[0] == 'element':
            in_vertex = t[1] == 'vertex'
            if in_vertex:
                count = int(t[2])
        elif t[0] == 'property' and in_vertex:
            props.append((t[-1], types[t[1]]))
    if fmt != 'binary_little_endian':
        raise RuntimeError(f'PLY 格式 {fmt} 不支持')
    dt = np.dtype([(n, '<' + t) for n, t in props])
    v = np.frombuffer(b, dt, count=count, offset=head_end)
    idx = np.sort(rng.choice(count, max_points, replace=False)) if count > max_points else np.arange(count)
    xyz = np.stack([v['x'][idx], v['y'][idx], v['z'][idx]], 1).astype(np.float32)
    if all(c in v.dtype.names for c in ('red', 'green', 'blue')):
        rgb = np.stack([v['red'][idx], v['green'][idx], v['blue'][idx]], 1).astype(np.uint8)
    else:
        rgb = np.full((len(idx), 3), 128, np.uint8)
    return xyz, rgb, int(count)


def find_root(tv, h):
    cands = [f'{h}/{c}' for c in PROJECT_CANDIDATES]
    cands += sorted(posixpath.dirname(n) for n in tv.m if n.startswith(f'{h}/') and n.endswith('/final_assets.json'))
    for c in cands:
        if tv.member(f'{c}/sparse/0/images.bin') is not None:
            return c
    return None


def depth_entries(tv, root, ims):
    """帧名 stem -> (深度 npz 的 archive 路径)。优先 final_assets.json 的 frames，其次 depth/ 目录。"""
    out = {}
    fa = tv.read(f'{root}/final_assets.json')
    fa = json.loads(fa) if fa else {}
    frames = fa.get('frames') or []
    if frames:
        named = all(any(k in f for k in ('name', 'image', 'image_name')) for f in frames)
        for i, f in enumerate(frames):
            if not f.get('depth'):
                continue
            if named:
                nm = f.get('name') or f.get('image_name') or f.get('image')
            elif len(frames) == len(ims):
                nm = ims[i]['name']             # 与 pda_assets.prepare_final 相同：frames 与按名字排序的注册图逐一对应
            else:
                continue
            out[stem(nm)] = posixpath.normpath(posixpath.join(root, f['depth']))
    if not out:
        files = [f for f in tv.listdir(f'{root}/depth') if f.endswith('.npz')]
        by_stem = {stem(f): f for f in files}
        if all(stem(im['name']) in by_stem for im in ims):
            out = {stem(im['name']): f'{root}/depth/{by_stem[stem(im["name"])]}' for im in ims}
        elif len(files) == len(ims):
            out = {stem(im['name']): f'{root}/depth/{f}' for im, f in zip(ims, sorted(files))}
    return out, fa


def run_new(args):
    io_ = IO()
    sids = read_scenes(args)
    os.makedirs(osp.join(args.out, 'new'), exist_ok=True)
    frames_json = load_frames_json(args.out)
    if args.frames_from:
        with open(args.frames_from) as fh:
            frames_json = {**json.load(fh), **frames_json}
    rows = load_ledger(args.ledger_dir) if args.ledger_dir else {}
    work = tempfile.mkdtemp(prefix='dl3dv_cmp_', dir=args.work)
    rng = np.random.default_rng(0)
    picked, status = {}, []
    try:
        for i, sid in enumerate(sids):
            sub, h = sid.split('/')
            dst = osp.join(args.out, 'new', safe(sid) + '.npz')
            if osp.exists(dst) and not args.force:
                status.append(dict(sid=sid, status='skip_existing'))
                continue
            t0 = time.time()
            local = osp.join(work, f'{h}.tar')
            try:
                src = (rows.get(sid) or {}).get('result')
                if not src:
                    if not args.results:
                        raise RuntimeError('没有 --results，台账里也没有 result 路径')
                    src = join(args.results, sub, f'{h}.tar')
                if args.results and osp.isdir(args.results) and not osp.exists(src):
                    alt = osp.join(args.results, f'{h}.tar')        # 本地目录也可以是平铺的 <hash>.tar
                    src = alt if osp.exists(alt) else src
                io_.copy_to_local(src, local)
                tv = TarView(local)
                root = find_root(tv, h)
                if root is None:
                    raise RuntimeError('tar 里没有最终工程（sparse/0/images.bin）')
                cams = read_cameras_bin(tv.read(f'{root}/sparse/0/cameras.bin'))
                ims = read_images_bin(tv.read(f'{root}/sparse/0/images.bin'))
                dmap, fa = depth_entries(tv, root, ims)
                for key, rel in (('final_assets', f'{root}/final_assets.json'), ('frame_sampling', f'{root}/frame_sampling.json'),
                                 ('cameras_bin', f'{root}/sparse/0/cameras.bin'), ('images_bin', f'{root}/sparse/0/images.bin')):
                    b = tv.read(rel)          # 原样留一份：格式是从代码推断的，真实数据回来后据此核对
                    if b:
                        ext = '.json' if key in ('final_assets', 'frame_sampling') else '.bin'
                        with open(osp.join(args.out, 'new', f'{safe(sid)}_{key}{ext}'), 'wb') as fh:
                            fh.write(b)
                if dmap:                      # 第一个深度 npz 的键/形状/dtype，核对格式用
                    with np.load(io.BytesIO(tv.read(next(iter(dmap.values()))))) as z:
                        npz_keys = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
                else:
                    npz_keys = None
                depth_dir = tv.listdir(f'{root}/depth')[:5]
                by = {stem(im['name']): im for im in ims}
                names, miss = pick_frames(sorted(by), sid, args, frames_json)
                picked[sid] = names + miss
                rgbs, depths, valids, Ks, w2cs, hw = [], [], [], [], [], []
                from PIL import Image
                for n in names:
                    im = by[n]
                    cam = cams[im['camera_id']]
                    if cam['model'] not in ('PINHOLE', 'SIMPLE_PINHOLE'):
                        raise RuntimeError(f"相机模型是 {cam['model']}，这里只处理去畸变后的针孔")
                    w2c = np.eye(4)
                    w2c[:3, :3], w2c[:3, 3] = qvec2rot(im['q']), im['t']
                    b = tv.read(f"{root}/images/{im['name']}")
                    rgb = np.array(Image.open(io.BytesIO(b)).convert('RGB')) if b else None
                    p = dmap.get(n)
                    db = tv.read(p) if p else None
                    if db is None:
                        d = np.full((cam['height'], cam['width']), np.nan, np.float32)
                        v = np.zeros_like(d, bool)
                    else:
                        with np.load(io.BytesIO(db)) as z:
                            d = np.asarray(z['depth'], np.float32)
                            v = np.asarray(z['valid_mask'], bool) if 'valid_mask' in z.files else np.ones(d.shape, bool)
                        v &= np.isfinite(d) & (d > 0)
                    K = cam_K(cam)
                    if d.shape != (cam['height'], cam['width']):        # 深度网格与相机尺寸不同：按比例换 K
                        sx, sy = d.shape[1] / cam['width'], d.shape[0] / cam['height']
                        K = np.diag([sx, sy, 1.0]) @ K
                    if rgb is None:
                        rgb = np.zeros((*d.shape, 3), np.uint8)
                    elif rgb.shape[:2] != d.shape:
                        import cv2
                        rgb = cv2.resize(rgb, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
                    rgbs.append(rgb), depths.append(d), valids.append(v), Ks.append(K), w2cs.append(w2c)
                    hw.append(d.shape)
                if not names:
                    raise RuntimeError(f'选中的帧都没有注册：{miss[:4]}')
                if len(set(hw)) > 1:          # 极少见：同一场景帧尺寸不一 -> 填到最大尺寸，记下真实尺寸
                    H, W = max(x[0] for x in hw), max(x[1] for x in hw)
                    pad = lambda a, fill: np.pad(a, [(0, H - a.shape[0]), (0, W - a.shape[1])] + [(0, 0)] * (a.ndim - 2),
                                                 constant_values=fill)
                    rgbs = [pad(a, 0) for a in rgbs]
                    depths = [pad(a, np.nan) for a in depths]
                    valids = [pad(a, False) for a in valids]
                all_names = sorted(by)
                all_w2c = []
                for n in all_names:
                    m = np.eye(4)
                    m[:3, :3], m[:3, 3] = qvec2rot(by[n]['q']), by[n]['t']
                    all_w2c.append(m)
                ply = tv.read(f'{root}/sparse/0/points3D.ply')
                if ply:
                    pxyz, prgb, pn = read_ply_xyzrgb(ply, args.max_ply_points, rng)
                else:
                    pxyz, prgb, pn = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), 0
                small = {}
                for key, rel in (('frame_sampling', f'{root}/frame_sampling.json'), ('status', f'{root}/../status.json'),
                                 ('result', f'{h}/project/result.json'), ('vlm', f'{h}/vlm.json'),
                                 ('ledger_row', f'{h}/ledger_row.json')):
                    b = tv.read(rel)
                    if b:
                        try:
                            small[key] = json.loads(b)
                        except ValueError:
                            pass
                st = small.get('status') or {}
                lr = small.get('ledger_row') or rows.get(sid) or {}
                info = dict(sid=sid, tar=src, root=root, missing=miss, n_registered=len(ims),
                            n_depth_files=len(dmap), hw=[list(x) for x in hw], ply_points=pn,
                            depth_npz_keys=npz_keys, depth_dir_head=depth_dir,
                            final_assets={k: fa.get(k) for k in ('format', 'status', 'alignment') if k in fa},
                            quality_status=st.get('quality_status') or (lr.get('glr') or {}).get('quality_status'),
                            ledger_class=lr.get('class'), vlm=_brief_vlm(small.get('vlm') or lr.get('vlm')),
                            frame_sampling={k: v for k, v in (small.get('frame_sampling') or {}).items()
                                            if not isinstance(v, (list, dict))})
                np.savez_compressed(dst + '.tmp.npz', names=np.array(names), rgb=np.stack(rgbs),
                                    depth=np.stack(depths).astype(np.float32), valid=np.stack(valids), K=np.stack(Ks),
                                    w2c=np.stack(w2cs), all_names=np.array(all_names), all_w2c=np.stack(all_w2c),
                                    ply_xyz=pxyz, ply_rgb=prgb, info=np.array(json.dumps(info, default=str)))
                os.replace(dst + '.tmp.npz', dst)
                status.append(dict(sid=sid, status='ok', frames=len(names), missing=len(miss), sec=round(time.time() - t0, 1)))
                log(f'new {i + 1}/{len(sids)} {sid} 帧 {len(names)}（缺 {len(miss)}）{time.time() - t0:.1f}s')
            except Exception as exc:  # noqa: BLE001
                status.append(dict(sid=sid, status='error', error=f'{type(exc).__name__}: {exc}'[:300]))
                log(f'new {i + 1}/{len(sids)} {sid} 失败：{type(exc).__name__}: {exc}')
            finally:
                if osp.exists(local):
                    os.remove(local)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    save_frames_json(args.out, picked)
    _write_status(args.out, 'new', status, getattr(args, 'shard', ''))


def _brief_vlm(v):
    if not isinstance(v, dict):
        return None
    return {k: v.get(k) for k in ('verdict', 'label', 'decision', 'answer', 'reason') if k in v} or None


def _write_status(out, what, status, shard=''):
    p = osp.join(out, f'status_{what}' + (f"_s{shard.replace('/', 'of')}" if shard else '') + '.json')
    with open(p, 'w') as fh:
        json.dump(status, fh, indent=1, ensure_ascii=False)
    c = {}
    for s in status:
        c[s['status']] = c.get(s['status'], 0) + 1
    log(f'{what} 完成：{c} -> {p}')


# ============================================================================ 3) 几何与指标
def umeyama(src, dst):
    """dst ≈ s R src + t（相机中心）。点太少或退化时返回 None。"""
    if len(src) < 3:
        return None
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    vs = (xs ** 2).sum() / len(src)
    if vs < 1e-12:
        return None
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / vs
    return s, R, md - s * R @ ms


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pose_check(old, new):
    """旧 c2w（WAI 世界）对新位姿（COLMAP 世界）：只用两边都有的帧。"""
    on = {str(n): i for i, n in enumerate(old['all_names'])}
    common = [str(n) for n in new['all_names'] if str(n) in on]
    out = dict(n_common=len(common))
    if len(common) < 3:
        return out, None
    Co = np.stack([old['all_c2w'][on[n]][:3, 3] for n in common])
    nn = {str(n): i for i, n in enumerate(new['all_names'])}
    c2w_new = np.stack([np.linalg.inv(new['all_w2c'][nn[n]]) for n in common])
    Cn = c2w_new[:, :3, 3]
    sim = umeyama(Co, Cn)
    if sim is None:
        return out, None
    s, R, t = sim
    path = float(np.linalg.norm(np.diff(Cn, axis=0), axis=1).sum())
    res = np.linalg.norm((s * (R @ Co.T)).T + t - Cn, axis=1)
    rots = [rot_deg((R @ old['all_c2w'][on[n]][:3, :3]).T @ c2w_new[k][:3, :3]) for k, n in enumerate(common)]
    out.update(sim3_scale=float(s), center_rmse_over_path=float(np.sqrt((res ** 2).mean()) / max(path, 1e-12)),
               rot_med_deg=float(np.median(rots)), rot_max_deg=float(np.max(rots)), path_new=path)
    return out, sim


def distort(xn, yn, k):
    k1, k2, p1, p2 = k
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn), yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn


def new_points_world(new, j):
    """新深度第 j 帧有效像素 -> 世界（COLMAP）坐标；返回 (Xw, z_cam)。像素中心按 COLMAP 约定（u+0.5）。"""
    d, v, K, w2c = new['depth'][j], new['valid'][j], new['K'][j], new['w2c'][j]
    ys, xs = np.nonzero(v)
    z = d[ys, xs].astype(np.float64)
    x = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    Xc = np.stack([x, y, z], 1)
    c2w = np.linalg.inv(w2c)
    return Xc @ c2w[:3, :3].T + c2w[:3, 3], z


def project_old(Xw_new, sim, c2w_old, K_old, hw, dist=None, pixel_center=0.5):
    """新世界点 -> 旧世界（Sim3 逆）-> 旧相机 -> 旧图像素；返回 z-buffer 后的稀疏深度图（旧单位）。"""
    s, R, t = sim
    Xo = ((Xw_new - t) @ R) / s                       # R^T (X - t) / s，按行
    w2c = np.linalg.inv(c2w_old)
    Xc = Xo @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    ok = z > 1e-9
    xn, yn = Xc[ok, 0] / z[ok], Xc[ok, 1] / z[ok]
    if dist is not None:
        xn, yn = distort(xn, yn, dist)
    u = K_old[0, 0] * xn + K_old[0, 2] - pixel_center
    v = K_old[1, 1] * yn + K_old[1, 2] - pixel_center
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    H, W = hw
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    zz = z[ok][inb]
    flat = vi[inb] * W + ui[inb]
    out = np.full(H * W, np.inf)
    np.minimum.at(out, flat, zz)
    out[~np.isfinite(out)] = 0
    return out.reshape(H, W)


def depth_stats(ref, est, m):
    """ref = 新（投影到旧图后），est = 旧深度；只用 m 像素。返回比值与对齐后误差。"""
    if m.sum() < 20:
        return None
    r, e = ref[m], est[m]
    ratio = r / e
    s = np.median(ratio)
    rel = np.abs(e * s - r) / r
    q = np.maximum(e * s / r, r / (e * s))
    return dict(n=int(m.sum()), ratio_med=float(s), absrel=float(rel.mean()), absrel_med=float(np.median(rel)),
                d105=float((q < 1.05).mean()), d110=float((q < 1.10).mean()), d125=float((q < 1.25).mean()))


def scene_metrics(old, new, sim):
    """逐帧：新深度投到旧图（针孔 / 带畸变），与旧深度（训练掩码内）比较。"""
    oinfo = json.loads(str(old['info']))
    cam = oinfo.get('camera') or {}
    dist = None
    if all(k in cam for k in ('k1', 'k2', 'p1', 'p2')):
        dist = tuple(float(cam[k]) for k in ('k1', 'k2', 'p1', 'p2'))
        if not any(dist):
            dist = None
    on = {str(n): i for i, n in enumerate(old['names'])}
    per, proj = [], {}
    hw = old['depth'].shape[1:3]
    for j, n in enumerate(map(str, new['names'])):
        if n not in on:
            continue
        i = on[n]
        Xw, _ = new_points_world(new, j)
        rec = dict(frame=n, new_valid=float(new['valid'][j].mean()), old_valid=float(old['mask'][i].mean()),
                   new_pts=int(len(Xw)))
        if sim is not None and len(Xw):
            for tag, dd in (('pin', None), ('dist', dist)):
                if tag == 'dist' and dd is None:
                    continue
                P = project_old(Xw, sim, old['c2w'][i], old['K'], hw, dd)
                m = (P > 0) & old['mask'][i]
                st = depth_stats(P, old['depth'][i].astype(np.float64), m)
                rec[tag] = st
                rec[f'{tag}_cover'] = float((P > 0).mean())
                proj[(n, tag)] = P
        per.append(rec)
    return per, proj, dist


def summarize_scene(sid, old, new, pose, per, dist):
    oinfo, ninfo = json.loads(str(old['info'])), json.loads(str(new['info']))
    cam = oinfo.get('camera') or {}
    rgb_hw = oinfo.get('rgb_hw')
    meta_wh = (cam.get('w'), cam.get('h'))
    row = dict(sid=sid, frames=len(per), quality=ninfo.get('quality_status'), vlm=json.dumps(ninfo.get('vlm'), ensure_ascii=False)
               if ninfo.get('vlm') else '', rgb_hw=rgb_hw, meta_wh=meta_wh,
               k_size_mismatch=bool(rgb_hw and meta_wh[0] and (int(meta_wh[0]) != rgb_hw[1] or int(meta_wh[1]) != rgb_hw[0])),
               has_dist=dist is not None, cam_model=cam.get('camera_model'))
    row.update({f'pose_{k}': v for k, v in pose.items()})
    for tag in ('pin', 'dist'):
        st = [p[tag] for p in per if p.get(tag)]
        if not st:
            continue
        ratios = np.array([s['ratio_med'] for s in st])
        row[f'{tag}_ratio_med'] = float(np.median(ratios))
        row[f'{tag}_ratio_spread'] = float(np.percentile(ratios, 90) / max(np.percentile(ratios, 10), 1e-12))
        for k in ('absrel', 'd105', 'd110', 'd125'):
            row[f'{tag}_{k}'] = float(np.median([s[k] for s in st]))
    row['old_valid'] = float(np.mean([p['old_valid'] for p in per])) if per else None
    row['new_valid'] = float(np.mean([p['new_valid'] for p in per])) if per else None
    if row.get('pin_absrel') is not None and row.get('dist_absrel') is not None:
        row['grid'] = 'distorted' if row['dist_absrel'] < 0.9 * row['pin_absrel'] else (
            'pinhole' if row['pin_absrel'] < 0.9 * row['dist_absrel'] else 'tie')
    return row


# ============================================================================ 3) 报告
BG_INVALID = (170, 170, 170)                  # 无值像素统一中灰，避免和白色（误差≈0）混淆


def colorize(d, valid, lo=None, hi=None, cmap='turbo', inv=False):
    """inv=True：按逆深度上色，近处暖色（红）、远处冷色（蓝），和常见深度图习惯一致。"""
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    out = np.zeros((*d.shape, 3), np.uint8) + np.array(BG_INVALID, np.uint8)
    valid = valid & np.isfinite(d) & ((d > 0) if inv else True)
    if valid.sum() == 0:
        return out, (0, 1)
    x = np.where(valid, 1.0 / np.maximum(d, 1e-9), 0) if inv else d
    v = x[valid]
    lo = np.percentile(v, 2) if lo is None else lo
    hi = np.percentile(v, 98) if hi is None else hi
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    out[valid] = (cm(t[valid])[:, :3] * 255).astype(np.uint8)
    return out, (lo, hi)


def dilate_sparse(img, valid, k=3):
    import cv2
    if k <= 1:
        return img, valid
    ker = np.ones((k, k), np.uint8)
    vd = cv2.dilate(valid.astype(np.uint8), ker) > 0
    # 用最近的有效像素填（最大池化近似）；只用于显示
    big = np.where(valid, img, -np.inf).astype(np.float32)
    big = cv2.dilate(big, ker)
    out = np.where(vd, big, 0)
    return out, vd


def dilate_signed(img, valid, k=3):
    """稀疏有符号图的显示膨胀：取邻域内 |值| 最大者，保留符号（只用于显示）。"""
    pos, vd = dilate_sparse(np.maximum(img, 0), valid, k)
    neg, _ = dilate_sparse(np.maximum(-img, 0), valid, k)
    return np.where(pos >= neg, pos, -neg), vd


def edge_overlay(rgb, depth, valid, color=(255, 0, 255)):
    """原色 RGB + 深度不连续边（品红）：看深度边是否贴着图像边。"""
    import cv2
    base = rgb[..., :3].astype(np.uint8).copy()
    ld = np.where(valid & (depth > 0), np.log(np.maximum(depth, 1e-6)), 0).astype(np.float32)
    gx = cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    vv = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    e = (mag / 8 > 0.05) & vv          # Sobel 增益约 8；相邻像素 log 深度跳 >5% 才算边，斜面不算
    base[e] = color
    return base


def render_scene_png(sid, old, new, per, proj, dest, z_far=60.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    on = {str(n): i for i, n in enumerate(old['names'])}
    nn = {str(n): j for j, n in enumerate(new['names'])}
    frames = [p['frame'] for p in per]
    cols = ['旧 RGB（images_8）+ 旧深度边（品红）', '旧深度（训练实际监督像素；近红远蓝，灰=无）', '旧置信度（黄高紫低；>0.25 为 conf 掩码）',
            '新 RGB（去畸变后原图）', '新深度（同色标，膨胀显示；灰=无）', '新/旧 偏差（红=新更远 蓝=新更近，±10%；白=一致）']
    oh, ow = old['depth'].shape[1:3]
    cw = 3.4                                   # 每列宽（英寸）；行高按旧图宽高比，外加一行文字
    rh = cw * oh / ow + 0.32
    top_pad, nrow, ncol = 0.75, len(frames), len(cols)
    fig_h = rh * nrow + top_pad
    fig, axes = plt.subplots(nrow, ncol, figsize=(cw * ncol, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=1 - top_pad / fig_h, wspace=0.02,
                        hspace=0.32 / (cw * oh / ow))
    for r, n in enumerate(frames):
        i, j = on[n], nn[n]
        rec = per[r]
        o_rgb, o_d, o_m, o_c = old['rgb'][i], old['depth'][i], old['mask'][i], old['conf'][i].astype(np.float32)
        n_rgb, n_d, n_v = new['rgb'][j], new['depth'][j], new['valid'][j]
        P = proj.get((n, 'pin'))
        st = rec.get('pin')
        s_scene = None
        if st:
            s_scene = st['ratio_med']
        o_tr = (o_d > 0) & (o_d < z_far)          # 训练实际监督的像素：conf 掩码在训练里没有生效
        od_c, rng_o = colorize(o_d, o_tr, inv=True)
        nd_show, nv_show = dilate_sparse(np.nan_to_num(n_d), n_v, 3)
        # 新深度用旧深度同一色标（换算到旧单位）：nd / s
        if s_scene:
            nd_c, _ = colorize(nd_show / s_scene, nv_show, *rng_o, inv=True)
        else:
            nd_c, _ = colorize(nd_show, nv_show, inv=True)
        err = np.zeros(o_d.shape, np.float32)
        em = np.zeros(o_d.shape, bool)
        if P is not None and s_scene:
            em = (P > 0) & o_tr
            err[em] = np.log(P[em] / (o_d[em] * s_scene))
        err_d, em_d = dilate_signed(err, em, 3)
        er_c, _ = colorize(err_d, em_d, -0.10, 0.10, 'RdBu_r')
        conf_c, _ = colorize(o_c, np.ones_like(o_m), 0, 1, 'viridis')
        imgs = [edge_overlay(o_rgb, o_d, o_m), od_c, conf_c, n_rgb[..., :3].astype(np.uint8), nd_c, er_c]
        for c, im in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(im, aspect='auto')
            ax.set_xticks([]), ax.set_yticks([])
        txt = f"{n}   旧训练像素 {o_tr.mean():.0%}（conf>0.25 {rec['old_valid']:.0%}）   新有效 {rec['new_valid']:.1%}"
        if st:
            txt += f"   新/旧 {st['ratio_med']:.3f}   conf 内 AbsRel {st['absrel']:.3f}   δ<1.05 {st['d105']:.0%}"
        d2 = rec.get('dist')
        if d2:
            txt += f"   conf 内 AbsRel(带畸变投影) {d2['absrel']:.3f}"
        axes[r, 0].text(0, 1.02, txt, transform=axes[r, 0].transAxes, fontsize=8, ha='left', va='bottom')
    for c, name in enumerate(cols):
        axes[0, c].text(0.5, 1.02 + 0.32 / (cw * oh / ow) * 0.9, name, transform=axes[0, c].transAxes, fontsize=9,
                        ha='center', va='bottom', weight='bold')
    fig.suptitle(sid, fontsize=10, y=1 - 0.12 / fig_h)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def _font_setup():
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('PingFang SC', 'Heiti SC', 'Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Zen Hei', 'SimHei',
                 'Source Han Sans SC', 'Arial Unicode MS'):
        if name in have:
            matplotlib.rcParams['font.sans-serif'] = [name] + matplotlib.rcParams['font.sans-serif']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


def old_points_world(old, i, sim, max_n, rng, z_far=60.0):
    """旧深度第 i 帧 -> 新（COLMAP）世界。取训练实际监督的像素（depth>0 且 <z_far，conf 掩码不生效），
    返回 (Xw, rgb, conf_ok)，conf_ok 标记 conf>0.25 的子集。"""
    d, m, K, c2w = old['depth'][i], old['mask'][i], old['K'], old['c2w'][i]
    ys, xs = np.nonzero((d > 0) & (d < z_far))
    if len(ys) > max_n:
        k = rng.choice(len(ys), max_n, replace=False)
        ys, xs = ys[k], xs[k]
    z = d[ys, xs].astype(np.float64)
    X = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    Xw = X @ c2w[:3, :3].T + c2w[:3, 3]
    s, R, t = sim
    return (s * (Xw @ R.T)) + t, old['rgb'][i][ys, xs, :3], m[ys, xs]


def _b64(a):
    import base64
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def render_scene_3d(sid, old, new, sim, dest, max_per_frame=6000):
    """左右两个 3D 视图（同步视角）：左 = 旧深度点（训练实际像素），右 = 新深度点；都在新（COLMAP）世界。
    颜色可切换：原图 RGB / 单色 / 按帧；旧点可切换只看 conf>0.25。点数据用 base64 嵌入，页面体积小。"""
    rng = np.random.default_rng(0)
    on = {str(n): i for i, n in enumerate(old['names'])}
    O = dict(xyz=[], rgb=[], conf=[], fid=[])
    N = dict(xyz=[], rgb=[], fid=[])
    frames = [str(n) for n in new['names']]
    for j, n in enumerate(frames):
        if n in on and sim is not None:
            Xw, rgb, ok = old_points_world(old, on[n], sim, max_per_frame, rng)
            O['xyz'].append(Xw); O['rgb'].append(rgb); O['conf'].append(ok); O['fid'].append(np.full(len(Xw), j))
        d, v = new['depth'][j], new['valid'][j]
        ys, xs = np.nonzero(v)
        Xw, _ = new_points_world(new, j)
        rgb = new['rgb'][j][ys, xs, :3]
        if len(Xw) > max_per_frame:
            k = rng.choice(len(Xw), max_per_frame, replace=False)
            Xw, rgb = Xw[k], rgb[k]
        N['xyz'].append(Xw); N['rgb'].append(rgb); N['fid'].append(np.full(len(Xw), j))

    def pack(D):
        if not D['xyz']:
            return dict(n=0)
        out = dict(n=int(sum(len(x) for x in D['xyz'])),
                   xyz=_b64(np.concatenate(D['xyz']).astype(np.float32)),
                   rgb=_b64(np.concatenate(D['rgb']).astype(np.uint8)),
                   fid=_b64(np.concatenate(D['fid']).astype(np.uint8)))
        if 'conf' in D:
            out['conf'] = _b64(np.concatenate(D['conf']).astype(np.uint8))
        return out

    cn = np.stack([np.linalg.inv(w)[:3, 3] for w in new['all_w2c']])
    # 场景的「上」方向：OpenCV 相机 -y 轴在世界里的平均方向
    up = -np.mean([np.linalg.inv(w)[:3, 1] for w in new['all_w2c']], 0)
    up = up / max(np.linalg.norm(up), 1e-9)
    co = None
    if sim is not None:
        s_, R_, t_ = sim
        co = ((s_ * (old['all_c2w'][:, :3, 3] @ R_.T)) + t_)
    data = dict(sid=sid, old=pack(O), new=pack(N), frames=frames, up=up.round(4).tolist(),
                cam_new=np.round(cn, 4).tolist(), cam_old=None if co is None else np.round(co, 4).tolist())
    page = _3D_TEMPLATE.replace('%%TITLE%%', html.escape(sid)).replace('%%DATA%%', json.dumps(data))
    with open(dest, 'w') as fh:
        fh.write(page)


_3D_TEMPLATE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%%TITLE%% 3D 对比</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#d9d9de;--chip:#f2f2f5;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;--fg:#ececf0;--mut:#9a9aa3;--line:#2c2c31;--chip:#1e1e22;--acc:#6ea0ff}}
:root[data-theme="dark"]{--bg:#121214;--fg:#ececf0;--mut:#9a9aa3;--line:#2c2c31;--chip:#1e1e22;--acc:#6ea0ff}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px/1.45 -apple-system,"PingFang SC","Noto Sans SC",sans-serif}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;padding:8px 16px;border-bottom:1px solid var(--line)}
header h1{font-size:14px;margin:0;font-weight:600;word-break:break-all}
.ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.ctl span{color:var(--mut)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;background:var(--chip);color:var(--fg);padding:4px 9px;font:inherit;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--acc);color:#fff}
input[type=range]{width:90px}
main{display:grid;grid-template-columns:1fr 1fr;height:calc(100% - 50px)}
.pane{position:relative;min-height:0;border-right:1px solid var(--line)}
.pane:last-child{border-right:0}
.pane h2{position:absolute;z-index:2;top:6px;left:10px;margin:0;font-size:13px;font-weight:600;background:color-mix(in srgb,var(--bg) 80%,transparent);padding:2px 6px;border-radius:4px}
.pane h2 small{font-weight:400;color:var(--mut);margin-left:6px}
.plot{position:absolute;inset:0}
@media (max-width:760px){main{grid-template-columns:1fr;grid-template-rows:1fr 1fr;height:auto}.pane{height:60vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<header><h1>%%TITLE%%</h1>
<div class="ctl"><span>颜色</span><div class="seg" id="cm">
<button data-v="rgb" aria-pressed="true">原图 RGB</button><button data-v="solid" aria-pressed="false">单色</button><button data-v="frame" aria-pressed="false">按帧</button></div></div>
<div class="ctl"><span>旧点</span><div class="seg" id="om">
<button data-v="train" aria-pressed="true">训练实际像素</button><button data-v="conf" aria-pressed="false">仅 conf&gt;0.25</button></div></div>
<div class="ctl"><span>点大小</span><input id="ps" type="range" min="0.5" max="4" step="0.25" value="1.5"></div>
<div class="ctl"><span>帧</span><select id="fr"><option value="-1">全部</option></select></div>
</header>
<main>
<div class="pane"><h2>旧标签 · mvsanywhere<small id="on"></small></h2><div class="plot" id="pL"></div></div>
<div class="pane"><h2>新标签 · GLR RoMa<small id="nn"></small></h2><div class="plot" id="pR"></div></div>
</main>
<script>
const D = %%DATA%%;
function dec(b64, T){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer)}
function unpack(P){if(!P.n)return null;const x=dec(P.xyz,Float32Array),c=dec(P.rgb,Uint8Array),f=dec(P.fid,Uint8Array);
  const o={n:P.n,X:new Float32Array(P.n),Y:new Float32Array(P.n),Z:new Float32Array(P.n),rgb:c,fid:f,conf:P.conf?dec(P.conf,Uint8Array):null};
  for(let i=0;i<P.n;i++){o.X[i]=x[3*i];o.Y[i]=x[3*i+1];o.Z[i]=x[3*i+2]}return o}
const OLD=unpack(D.old), NEW=unpack(D.new);
const st={cm:'rgb',om:'train',ps:1.5,fr:-1};
const fsel=document.getElementById('fr');D.frames.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n;fsel.appendChild(o)});
function frameColor(i,n){const h=(i/Math.max(n,1))*300;return `hsl(${h},75%,50%)`}
function build(P,solid,isOld){
  if(!P)return {x:[],y:[],z:[],c:[],k:0};
  const x=[],y=[],z=[],c=[];
  for(let i=0;i<P.n;i++){
    if(st.fr>=0&&P.fid[i]!=st.fr)continue;
    if(isOld&&st.om==='conf'&&P.conf&&!P.conf[i])continue;
    x.push(P.X[i]);y.push(P.Y[i]);z.push(P.Z[i]);
    c.push(st.cm==='rgb'?`rgb(${P.rgb[3*i]},${P.rgb[3*i+1]},${P.rgb[3*i+2]})`:st.cm==='solid'?solid:frameColor(P.fid[i],D.frames.length));
  }return {x,y,z,c,k:x.length}}
function cams(){const t=[];const L=(p,name,col,dash)=>p&&t.push({type:'scatter3d',mode:'lines',name,x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),line:{color:col,width:3,dash},hoverinfo:'name',showlegend:false});
  L(D.cam_new,'新相机轨迹','#2563eb','solid');L(D.cam_old,'旧相机轨迹','#d97706','dash');return t}
// 两侧用同一坐标范围，保证同步视角时比例一致
function range(){let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];for(const P of [OLD,NEW]){if(!P)continue;const A=[P.X,P.Y,P.Z];
  for(let a=0;a<3;a++){const s=Array.from(A[a]).sort((u,v)=>u-v);lo[a]=Math.min(lo[a],s[Math.floor(s.length*.01)]);hi[a]=Math.max(hi[a],s[Math.floor(s.length*.99)])}}
  return [lo,hi]}
const [LO,HI]=range();const span=Math.max(HI[0]-LO[0],HI[1]-LO[1],HI[2]-LO[2]);
function ax(a){const m=(LO[a]+HI[a])/2;return {range:[m-span*.55,m+span*.55],visible:false}}
const layout=()=>({margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:getComputedStyle(document.body).color},
  scene:{xaxis:ax(0),yaxis:ax(1),zaxis:ax(2),aspectmode:'cube',camera:{up:{x:D.up[0],y:D.up[1],z:D.up[2]}}},showlegend:false,uirevision:'keep'});
function draw(){
  const o=build(OLD,'#d97706',true),n=build(NEW,'#2563eb',false);
  document.getElementById('on').textContent=`${o.k.toLocaleString()} 点`;document.getElementById('nn').textContent=`${n.k.toLocaleString()} 点`;
  const tr=(b,name)=>({type:'scatter3d',mode:'markers',name,x:b.x,y:b.y,z:b.z,marker:{size:st.ps,color:b.c,opacity:.9},hoverinfo:'skip'});
  Plotly.react('pL',[tr(o,'旧'),...cams()],layout(),{displaylogo:false,responsive:true});
  Plotly.react('pR',[tr(n,'新'),...cams()],layout(),{displaylogo:false,responsive:true});
}
draw();
// 同步视角：拖动一侧，另一侧跟随
let lock=false;
function sync(src,dst){document.getElementById(src).on('plotly_relayout',e=>{if(lock)return;const cam=e['scene.camera'];if(!cam)return;
  lock=true;Plotly.relayout(dst,{'scene.camera':cam}).then(()=>{lock=false})})}
sync('pL','pR');sync('pR','pL');
function seg(id,key){document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=()=>{document.querySelectorAll(`#${id} button`).forEach(x=>x.setAttribute('aria-pressed',x===b));st[key]=b.dataset.v;draw()})}
seg('cm','cm');seg('om','om');
document.getElementById('ps').oninput=e=>{st.ps=+e.target.value;draw()};
fsel.onchange=e=>{st.fr=+e.target.value;draw()};
</script></body></html>"""


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ''
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def run_report(args):
    # macOS 上 numpy 2.x + Accelerate 的 matmul 会报假的 divide/overflow 警告（结果正确）；无效值都已显式过滤
    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    _font_setup()
    rep = osp.join(args.out, 'report')
    os.makedirs(rep, exist_ok=True)
    olds = {f[:-4] for f in os.listdir(osp.join(args.out, 'old')) if f.endswith('.npz')} if osp.isdir(osp.join(args.out, 'old')) else set()
    news = {f[:-4] for f in os.listdir(osp.join(args.out, 'new')) if f.endswith('.npz')} if osp.isdir(osp.join(args.out, 'new')) else set()
    both = sorted(olds & news)
    log(f'旧 {len(olds)} 个，新 {len(news)} 个，两边都有 {len(both)} 个')
    if not both:
        raise SystemExit('没有两边都有的场景')
    rows, frames_rows = [], []
    for k, name in enumerate(both):
        old = dict(np.load(osp.join(args.out, 'old', name + '.npz')))
        new = dict(np.load(osp.join(args.out, 'new', name + '.npz')))
        sid = json.loads(str(new['info']))['sid']
        pose, sim = pose_check(old, new)
        per, proj, dist = scene_metrics(old, new, sim)
        row = summarize_scene(sid, old, new, pose, per, dist)
        row['page'] = name
        rows.append(row)
        for p in per:
            fr = dict(sid=sid, frame=p['frame'], old_valid=p['old_valid'], new_valid=p['new_valid'])
            for tag in ('pin', 'dist'):
                for kk, vv in (p.get(tag) or {}).items():
                    fr[f'{tag}_{kk}'] = vv
            frames_rows.append(fr)
        if per:
            render_scene_png(sid, old, new, per, proj, osp.join(rep, name + '.png'))
            render_scene_3d(sid, old, new, sim, osp.join(rep, name + '_3d.html'))
        log(f'report {k + 1}/{len(both)} {sid} 帧 {len(per)}  位姿旋转残差 {fmt(pose.get("rot_med_deg"))}°  '
            f'新/旧 {fmt(row.get("pin_ratio_med"))}  AbsRel {fmt(row.get("pin_absrel"))}')
    _write_csv(osp.join(rep, 'scenes.csv'), rows)
    _write_csv(osp.join(rep, 'frames.csv'), frames_rows)
    _write_index(rep, rows)
    log(f'报告：{osp.join(rep, "index.html")}')


def _write_csv(p, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()})


def _write_index(rep, rows):
    def med(key):
        v = [r[key] for r in rows if isinstance(r.get(key), float)]
        return float(np.median(v)) if v else None

    cols = [('sid', '场景', None), ('frames', '帧', None), ('quality', 'GLR 质量', None),
            ('pose_rot_med_deg', '位姿旋转残差°', 3), ('pose_center_rmse_over_path', '中心残差/路程', 4),
            ('pose_sim3_scale', 'Sim3 尺度', 3), ('pin_ratio_med', '新/旧 深度比', 3), ('pin_ratio_spread', '帧间比值 p90/p10', 3),
            ('pin_absrel', 'AbsRel（针孔）', 3), ('pin_d105', 'δ<1.05', 2), ('pin_d125', 'δ<1.25', 2),
            ('dist_absrel', 'AbsRel（带畸变）', 3), ('grid', '旧深度所在网格', None), ('old_valid', '旧有效像素', 2),
            ('new_valid', '新有效像素', 3), ('k_size_mismatch', 'K 尺寸≠RGB', None)]
    head = ''.join(f'<th>{html.escape(c[1])}</th>' for c in cols)
    body = []
    for r in rows:
        tds = []
        for key, _, nd in cols:
            v = r.get(key)
            if key == 'sid':
                tds.append(f'<td class="l"><a href="{r["page"]}.png">{html.escape(str(v))}</a> · '
                           f'<a href="{r["page"]}_3d.html">3D</a></td>')
            else:
                tds.append(f'<td>{html.escape(fmt(v, nd) if nd else fmt(v))}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    summ = (f"场景 {len(rows)} 个；中位数：位姿旋转残差 {fmt(med('pose_rot_med_deg'))}°，新/旧深度比 {fmt(med('pin_ratio_med'))}"
            f"（Sim3 尺度 {fmt(med('pose_sim3_scale'))}），帧间比值 p90/p10 {fmt(med('pin_ratio_spread'))}，"
            f"AbsRel 针孔 {fmt(med('pin_absrel'))} / 带畸变 {fmt(med('dist_absrel'))}，δ<1.05 {fmt(med('pin_d105'), 2)}，"
            f"旧有效像素 {fmt(med('old_valid'), 2)}，新有效像素 {fmt(med('new_valid'), 3)}")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>DL3DV 新旧标签对比</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:16px;color:#222}}
table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ccc;padding:3px 6px;text-align:right}}
td.l{{text-align:left}}th{{background:#f3f3f3;cursor:pointer;position:sticky;top:0}}
.note{{font-size:13px;line-height:1.6;max-width:1100px}}
</style></head><body>
<h2>DL3DV：Pi3 训练原标签（DL3DV_WAI）vs GLR-v3 新打标</h2>
<p class="note">{html.escape(summ)}</p>
<p class="note">怎么读：<b>位姿旋转残差</b>应接近 0（known-poses 用的是同一份 transforms.json，不为 0 说明约定没对上）。
<b>新/旧深度比</b>：新深度与位姿同为 SfM 单位，所以它应等于 Sim3 尺度（通常为 1）；偏离说明旧深度与旧位姿尺度不一致，
<b>帧间比值 p90/p10</b> 越接近 1 越好，大于 1.1 说明旧深度逐帧尺度不稳（多视图训练会互相矛盾）。
<b>AbsRel / δ</b> 是按每帧中位比对齐后、只在新深度有观测且旧训练掩码内的像素上算的。
<b>旧深度所在网格</b>：把新点按「训练假设的针孔」和「带畸变」两种方式投进旧图，哪种与旧深度更吻合（相差 10% 以上才判）；
训练里旧 RGB 是带畸变的 images_8、内参按针孔用，若旧深度在去畸变网格上，RGB 与深度在图像边缘会错位。
新深度是 RoMa 观测投影的稀疏深度（未做 PDA 补全），有效像素少是正常的。点表头可排序。</p>
<table id="t"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
  const tb=document.querySelector('#t tbody');const rs=[...tb.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
  rs.sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;const fx=parseFloat(x),fy=parseFloat(y);
    const c=(isNaN(fx)||isNaN(fy))?x.localeCompare(y):fx-fy;return asc?c:-c}});rs.forEach(r=>tb.appendChild(r))}});
</script></body></html>"""
    with open(osp.join(rep, 'index.html'), 'w') as fh:
        fh.write(page)


# ============================================================================ 4) 批量扫描：旧标签疑似失败、新标签可靠
# 每项指标除以阈值，>=1 记一个标签；场景分数 = 各项的最大值。阈值是按 6 场景试点定的起点，扫完看分布再调。
SCAN_RULES = (
    # key            说明                                          阈值
    ('absrel',      '训练像素 AbsRel 高（旧 vs 新）',                0.05),
    ('bad10',       '训练像素误差>10% 的比例高',                     0.12),
    ('bad_frames',  '坏帧比例高（单帧 AbsRel>0.08 或 >10% 占 20%+）', 0.25),
    ('scale',       '整体尺度偏（|log 新/旧| 中位）',                 0.05),
    ('drift',       '帧间尺度漂移（log p90/p10）',                    math.log(1.10)),
    ('mv',          '旧深度自身多视图前方违例（投到相邻帧后比那帧表面近 5%+ 的比例；遮挡解释不了）', 0.10),
)
SCAN_GATE = dict(new_cov=0.04, frames_ok=0.6, min_px=200)   # 新标签可靠的最低要求


def old_multiview(old, z_far):
    """旧深度第 i 帧反投到它的相邻帧（old 步 --neighbor-offset 读的那帧），和相邻帧旧深度比：不依赖新标签的自洽检查。"""
    if 'nb_depth' not in old:
        return None
    K = old['K']
    H, W = old['depth'].shape[1:]
    rs, ovs = [], []
    for i in range(len(old['names'])):
        if not str(old['nb_names'][i]):
            continue
        d = old['depth'][i].astype(np.float64)
        ys, xs = np.nonzero((d > 0) & (d < z_far))
        if len(ys) < 200:
            continue
        z = d[ys, xs]
        X = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        Xw = X @ old['c2w'][i][:3, :3].T + old['c2w'][i][:3, 3]
        w2c = np.linalg.inv(old['nb_c2w'][i])
        Xc = Xw @ w2c[:3, :3].T + w2c[:3, 3]
        zz = Xc[:, 2]
        ok = zz > 1e-6
        u = np.full(len(zz), -1, np.int64)
        v = np.full(len(zz), -1, np.int64)
        u[ok] = np.round(K[0, 0] * Xc[ok, 0] / zz[ok] + K[0, 2] - 0.5).astype(np.int64)
        v[ok] = np.round(K[1, 1] * Xc[ok, 1] / zz[ok] + K[1, 2] - 0.5).astype(np.int64)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dn = np.zeros(len(zz))
        dn[inb] = old['nb_depth'][i][v[inb], u[inb]].astype(np.float64)
        sel = inb & (dn > 0) & (dn < z_far)
        ovs.append(sel.mean())
        if sel.sum() >= 200:
            rs.append((zz[sel] - dn[sel]) / dn[sel])
    if not rs:
        return dict(mv_pairs=0)
    r = np.concatenate(rs)
    # 双向 |r|>5% 会把遮挡（点跑到相邻帧表面后面）也算进去；前方违例（点跑到表面前面）遮挡解释不了，用它打分
    return dict(mv_pairs=len(rs), mv_overlap=float(np.mean(ovs)), mv_med=float(np.median(np.abs(r))),
                mv_bad5=float(np.mean(np.abs(r) > 0.05)), mv_front5=float(np.mean(r < -0.05)))


def scan_scene(old, new, z_far):
    pose, sim = pose_check(old, new)
    row = dict(pose_rot_med_deg=pose.get('rot_med_deg'), pose_sim3_scale=pose.get('sim3_scale'))
    if sim is None:
        row['error'] = 'pose_check 失败（共同帧 <3）'
        return row, []
    on = {str(n): i for i, n in enumerate(old['names'])}
    H, W = old['depth'].shape[1:]
    frames, po, pn, pin_o, pin_n, pout_o, pout_n = [], [], [], [], [], [], []
    for j, n in enumerate(map(str, new['names'])):
        if n not in on:
            continue
        i = on[n]
        Xw, _ = new_points_world(new, j)
        P = project_old(Xw, sim, old['c2w'][i], old['K'], (H, W), None) if len(Xw) else np.zeros((H, W))
        od = old['depth'][i].astype(np.float64)
        tr = (od > 0) & (od < z_far)
        om = old['mask'][i]
        m = (P > 0) & tr
        rec = dict(frame=n, new_valid=float(new['valid'][j].mean()), tr_cov=float(tr.mean()), conf_cov=float(om.mean()),
                   n_px=int(m.sum()))
        if m.sum() >= SCAN_GATE['min_px']:
            rel = np.abs(od[m] - P[m]) / P[m]
            rec.update(ratio=float(np.median(P[m] / od[m])), absrel=float(rel.mean()), bad10=float(np.mean(rel > 0.1)))
            rec['bad'] = bool(rec['absrel'] > 0.08 or rec['bad10'] > 0.20)
            po.append(od[m]), pn.append(P[m])
            mi, mo = m & om, m & ~om
            pin_o.append(od[mi]), pin_n.append(P[mi]), pout_o.append(od[mo]), pout_n.append(P[mo])
        frames.append(rec)
    good = [f for f in frames if 'absrel' in f]
    row.update(frames=len(frames), frames_ok=len(good),
               new_cov=float(np.median([f['new_valid'] for f in frames])) if frames else 0.0,
               conf_cov=float(np.mean([f['conf_cov'] for f in frames])) if frames else None,
               tr_cov=float(np.mean([f['tr_cov'] for f in frames])) if frames else None)
    if good:
        o, n_ = np.concatenate(po), np.concatenate(pn)
        rel = np.abs(o - n_) / n_
        ratios = np.array([f['ratio'] for f in good])
        row.update(absrel=float(rel.mean()), bad10=float(np.mean(rel > 0.1)),
                   bad_frames=float(np.mean([f['bad'] for f in good])),
                   worst_frame_absrel=float(max(f['absrel'] for f in good)),
                   ratio_med=float(np.median(ratios)),
                   scale=float(abs(np.log(np.median(ratios)))),
                   drift=float(np.log(np.percentile(ratios, 90) / max(np.percentile(ratios, 10), 1e-9))) if len(ratios) >= 3 else 0.0)
        for tag, a, b in (('in_conf', pin_o, pin_n), ('low_conf', pout_o, pout_n)):
            a, b = np.concatenate(a), np.concatenate(b)
            if len(b) >= 200:
                r2 = np.abs(a - b) / b
                row[f'absrel_{tag}'] = float(r2.mean())
                row[f'bad10_{tag}'] = float(np.mean(r2 > 0.1))
    mv = old_multiview(old, z_far)
    if mv:
        row.update(mv)
        if mv.get('mv_pairs'):
            row['mv'] = mv['mv_front5']
    # 打分
    tags, score = [], 0.0
    for key, _, th in SCAN_RULES:
        v = row.get(key)
        if v is None or not math.isfinite(v):
            continue
        x = v / th
        row[f'x_{key}'] = round(x, 3)
        score = max(score, x)
        if x >= 1:
            tags.append(key)
    row['score'] = round(score, 3)
    row['tags'] = ','.join(tags)
    row['new_reliable'] = bool(row['new_cov'] >= SCAN_GATE['new_cov'] and frames and
                               row['frames_ok'] >= SCAN_GATE['frames_ok'] * len(frames))
    if tags and row['new_reliable']:
        strong = 'mv' in tags and any(t in tags for t in ('absrel', 'bad10', 'bad_frames', 'scale'))
        row['category'] = '旧疑似失败·新可靠' + ('（强）' if strong else '')
    elif tags:
        row['category'] = '有分歧·新标签覆盖不足'
    elif row['new_reliable']:
        row['category'] = '一致'
    else:
        row['category'] = '新标签覆盖不足'
    return row, frames


def run_scan(args):
    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    _font_setup()
    out = osp.join(args.out, 'scan')
    for d in ('panels', '3d'):
        os.makedirs(osp.join(out, d), exist_ok=True)
    ls = lambda d: {f[:-4] for f in os.listdir(osp.join(args.out, d)) if f.endswith('.npz') and not f.endswith('.tmp.npz')} \
        if osp.isdir(osp.join(args.out, d)) else set()
    olds, news = ls('old'), ls('new')

    def statuses(what):
        st = {}
        for f in sorted(os.listdir(args.out)):
            if f.startswith(f'status_{what}') and f.endswith('.json'):
                for r in json.load(open(osp.join(args.out, f))):
                    if r.get('status') != 'skip_existing' or r['sid'] not in st:
                        st[r['sid']] = r
        return st
    st_old, st_new = statuses('old'), statuses('new')
    log(f'旧 npz {len(olds)}，新 npz {len(news)}，两边都有 {len(olds & news)}')
    rows, frows = [], []
    for k, name in enumerate(sorted(news)):
        new = dict(np.load(osp.join(args.out, 'new', name + '.npz')))
        ninfo = json.loads(str(new['info']))
        sid = ninfo['sid']
        base = dict(sid=sid, name=name, quality=ninfo.get('quality_status'), n_registered=ninfo.get('n_registered'))
        if name not in olds:
            e = (st_old.get(sid) or {}).get('error', '没有旧 npz')
            rows.append({**base, 'category': '旧标签缺失/读取失败', 'score': None, 'tags': 'old_missing', 'error': e})
            continue
        old = dict(np.load(osp.join(args.out, 'old', name + '.npz')))
        try:
            row, frames = scan_scene(old, new, args.z_far)
        except Exception as exc:  # noqa: BLE001
            row, frames = dict(category='扫描出错', error=f'{type(exc).__name__}: {exc}'[:200]), []
        rows.append({**base, **row})
        for f in frames:
            frows.append(dict(sid=sid, name=name, **f))
        if (k + 1) % 20 == 0:
            log(f'scan {k + 1}/{len(news)}')
    for sid, r in st_new.items():          # 新标签抽取失败的（多半是 GLR 没打完）
        if r.get('status') == 'error' and safe(sid) not in news:
            rows.append(dict(sid=sid, name=safe(sid), category='新标签缺失（GLR 未完成或抽取失败）', score=None,
                             tags='new_missing', error=r.get('error')))
    rank = sorted(rows, key=lambda r: (-(r['score'] or -1)))
    keys = []
    for r in rank:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with open(osp.join(out, 'scan.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rank:
            w.writerow({kk: fmt(r.get(kk)) if isinstance(r.get(kk), float) else r.get(kk) for kk in keys})
    if frows:
        fk = []
        for r in frows:
            for kk in r:
                if kk not in fk:
                    fk.append(kk)
        with open(osp.join(out, 'scan_frames.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fk)
            w.writeheader()
            for r in frows:
                w.writerow({kk: fmt(r.get(kk)) if isinstance(r.get(kk), float) else r.get(kk) for kk in fk})
    cats = {}
    for r in rows:
        cats[r['category']] = cats.get(r['category'], 0) + 1
    rule_hits = {key: sum(1 for r in rows if key in (r.get('tags') or '').split(',')) for key, _, _ in SCAN_RULES}
    summary = dict(n=len(rows), categories=cats, rule_hits=rule_hits, rules=[dict(key=a, desc=b, thr=c) for a, b, c in SCAN_RULES],
                   gate=SCAN_GATE, z_far=args.z_far)
    with open(osp.join(out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    log(f'分类：{cats}')
    log(f'各规则命中：{rule_hits}')
    # 出图：疑似旧失败的前 K 个 + 分数最低的几个作对照
    flagged = [r for r in rank if str(r.get('category', '')).startswith('旧疑似失败')][:args.top]
    calm = [r for r in sorted((r for r in rows if r.get('category') == '一致'), key=lambda r: r['score'])][:args.controls]
    for r in flagged + calm:
        name = r['name']
        try:
            old = dict(np.load(osp.join(args.out, 'old', name + '.npz')))
            new = dict(np.load(osp.join(args.out, 'new', name + '.npz')))
            pose, sim = pose_check(old, new)
            per, proj, _ = scene_metrics(old, new, sim)
            png = osp.join(out, 'panels', name + '.png')
            render_scene_png(r['sid'], old, new, per, proj, png)
            try:
                from PIL import Image
                im = Image.open(png).convert('RGB')
                w_, h_ = im.size
                im.resize((1700, round(h_ * 1700 / w_)), Image.LANCZOS).save(png[:-4] + '.jpg', quality=82, optimize=True)
                os.remove(png)
            except Exception:  # noqa: BLE001
                pass
            render_scene_3d(r['sid'], old, new, sim, osp.join(out, '3d', name + '.html'), max_per_frame=4000)
            log(f"出图 {name}  {r.get('category')}  score={r.get('score')}  {r.get('tags')}")
        except Exception as exc:  # noqa: BLE001
            log(f'出图失败 {name}: {type(exc).__name__}: {exc}')
    with open(osp.join(out, 'rendered.json'), 'w') as fh:
        json.dump(dict(flagged=[r['name'] for r in flagged], controls=[r['name'] for r in calm]), fh, indent=1)
    log(f'扫描结果：{out}/scan.csv（{len(rows)} 行），出图 {len(flagged)} + 对照 {len(calm)}')


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)

    def common(p, scenes=True):
        p.add_argument('--out', required=True, help='输出目录（old/ new/ report/ frames.json）')
        if scenes:
            p.add_argument('--scenes', nargs='*', help='场景清单文件，每行 1K/<hash>')
            p.add_argument('--sids', nargs='*', help='直接给场景名')
            p.add_argument('--limit', type=int, default=0, help='从清单里均匀取这么多个场景（0=全部）')
            p.add_argument('--n-frames', type=int, default=8, help='每场景均匀抽几帧（frames.json 里已有选择时不用）')
            p.add_argument('--frames', default='', help='指定帧名，逗号分隔（如 frame_00001,frame_00100），所有场景通用')
            p.add_argument('--frames-from', default='', help='另一台机器写的 frames.json，保证两边抽同一组帧')
            p.add_argument('--force', action='store_true', help='已有 npz 也重抽')
            p.add_argument('--shard', default='', help='i/n：只处理场景清单的第 i 份（共 n 份），多进程并行用')

    def old_args(p):
        p.add_argument('--wai-root', default=WAI_ROOT)
        p.add_argument('--rgb-root', default=RGB_ROOT)
        p.add_argument('--rgb-subdir', default='images_8')
        p.add_argument('--rgb-ext', default='.png')
        p.add_argument('--conf-thres', type=float, default=0.25)
        p.add_argument('--depth-thres', type=float, default=1e-4)
        p.add_argument('--depth-interp', default='lanczos', choices=('lanczos', 'nearest'))
        p.add_argument('--no-pct-twice', dest='pct_twice', action='store_false')
        p.add_argument('--apply-trans', action='store_true', help='与 DL3DVDataset(apply_trans=True) 一致（配置里默认 False）')
        p.add_argument('--allow-local-io', action='store_true', help='训练仓库导入失败时也继续（只能读本地文件）')
        p.add_argument('--neighbor-offset', type=int, default=0, help='>0 时每个抽样帧再读第 +k 帧的旧深度，供 scan 查旧深度自身的多视图一致性')

    def new_args(p):
        p.add_argument('--results', default='', help='结果 tar 前缀：obs://.../<run>/results 或本地目录（<子集>/<hash>.tar 或平铺）')
        p.add_argument('--ledger-dir', default='', help='E/roma/runs/<run>：用台账里的 result 路径；没给 --scenes 时取全部 complete')
        p.add_argument('--work', default=None, help='下载 tar 的临时目录（每个约 1 GB，用完即删）')
        p.add_argument('--max-ply-points', type=int, default=40000)

    p = sp.add_parser('old', help='抽旧标签（DL3DV_WAI，与 DL3DVDataset 一致）')
    common(p), old_args(p)
    p = sp.add_parser('new', help='抽新标签（GLR-v3 结果 tar）')
    common(p), new_args(p)
    p = sp.add_parser('report', help='对比报告')
    common(p, scenes=False)
    p = sp.add_parser('scan', help='批量扫描：给每个场景打「旧标签疑似失败、新标签可靠」的分，排序并为前 K 个出图')
    common(p, scenes=False)
    p.add_argument('--top', type=int, default=24, help='为分数最高的前 K 个场景出 12 帧对比图和 3D 页')
    p.add_argument('--controls', type=int, default=4, help='再为分数最低的几个场景出图，作对照')
    p.add_argument('--z-far', type=float, default=60.0, help='与训练配置一致（datamix z_far）')
    p = sp.add_parser('all', help='old + new + report（一台机器两边都能读时）')
    common(p), old_args(p), new_args(p)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == 'old':
        run_old(args)
    elif args.cmd == 'new':
        run_new(args)
    elif args.cmd == 'report':
        run_report(args)
    elif args.cmd == 'scan':
        run_scan(args)
    else:
        run_old(args)
        args.frames_from = ''
        run_new(args)
        run_report(args)


if __name__ == '__main__':
    main()
