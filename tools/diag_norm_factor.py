#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断：复刻 Pi3Loss.prepare_gt 的归一化逻辑，对比【场景:ASE】vs【物体:CO3D】，
验证 norm_factor = "所有有效点到第一帧相机光心距离的均值" 是否把物体压扁。

Pi3Loss 里的归一化（关键）：
    w2c_target = se3_inverse(poses[:, 0])          # 转到第一帧相机系
    all_dis = all_pts.norm(dim=-1)                 # 点到第一帧相机原点的距离
    norm_factor = all_dis.sum() / valid_count      # 均值
    gt_pts /= norm_factor

输出列：
  norm_factor_mean        : 现有实现用的分母（mean）
  norm_factor_median      : 改用 median 会是多少（看 mean 被远点拉高多少）
  obj_extent_p90          : 点云中心化后 p90 半径 = 数据"自身几何尺度"
  compress_ratio          : obj_extent_p90 / norm_factor_mean
        归一化后"自身几何"占的动态范围；越小=被压越扁。
        场景(ASE) 期望 ~0.5-1.0；物体(CO3D) 若 <<0.3 即坐实被压扁。
  first_frame_sensitivity : 换不同帧当第一帧时 norm_factor 的相对波动(std/mean)

运行：python diag_norm_factor.py   （按实际改 import 路径与 data_root）
"""
import sys, os
# 脚本在 Pi3/tools/ 下，父目录 Pi3/ 是工程根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from datasets.ase_dataset import ASEDataset
from datasets.co3d_dataset import CO3DDataset

# ---- 数据集构建：shuffle=False 固定第一帧，保证同口径可复现 ----
DATASETS = {
    "ASE(场景)": lambda: ASEDataset(
        data_root="/efs-m3ga/c50041401/datasets/reliev3r_synth_data/ASE", 
        mode="train", resolution=[[518, 336]], frame_num=8,
        max_distance=8, split_ratio=0.9, verbose=False, shuffle=False,
    ),
    "CO3D(物体)": lambda: CO3DDataset(
        data_root="/efs-m3ga/z00533370/reliev3r_data/co3d",
        mode="train", resolution=[[518, 336]], frame_num=8,
        max_distance=8, split_ratio=1, min_num_images=24,
        verbose=False, shuffle=False, load_depth=True,
    ),
}

N_SAMPLES = 30
SEED = 0


def collect_world_points(views):
    """[(pts(M,3) world, cam_center(3)), ...]，用 pts3d + valid_mask + camera_pose(c2w) 平移。"""
    out = []
    for v in views:
        if "pts3d" not in v or "valid_mask" not in v or "camera_pose" not in v:
            continue
        pts = np.asarray(v["pts3d"], dtype=np.float64)
        m = np.asarray(v["valid_mask"], dtype=bool)
        pose = np.asarray(v["camera_pose"], dtype=np.float64)
        p = pts[m]
        if p.shape[0] == 0:
            continue
        out.append((p, pose[:3, 3]))
    return out


def norm_factor_for_first(vp, first_idx):
    """复刻 loss：所有点平移到 first 帧相机光心，算到原点距离的 mean / median。"""
    _, c0 = vp[first_idx]
    dists = [np.linalg.norm(p - c0, axis=-1) for p, _ in vp]
    d = np.concatenate(dists)
    return float(d.mean()), float(np.median(d))


def analyze(views):
    vp = collect_world_points(views)
    if len(vp) < 2:
        return None
    allp = np.concatenate([p for p, _ in vp], axis=0)
    center = np.median(allp, axis=0)
    extent = float(np.percentile(np.linalg.norm(allp - center, axis=-1), 90))
    nf_mean0, nf_med0 = norm_factor_for_first(vp, 0)
    nfs = [norm_factor_for_first(vp, i)[0] for i in range(len(vp))]
    sens = float(np.std(nfs) / (np.mean(nfs) + 1e-8))
    return {
        "norm_factor_mean":        nf_mean0,
        "norm_factor_median":      nf_med0,
        "obj_extent_p90":          extent,
        "compress_ratio":          extent / (nf_mean0 + 1e-8),
        "first_frame_sensitivity": sens,
    }


def _iter(idxs, name):
    try:
        from tqdm import tqdm
        return tqdm(idxs, desc=name)
    except Exception:
        return idxs


def run(name, build_fn, n=N_SAMPLES, seed=SEED):
    ds = build_fn()
    if len(ds) == 0:
        print(f"[{name}] empty dataset"); return
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    rows = []
    for i in _iter(idxs, name):
        try:
            r = analyze(ds[int(i)])
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  [{name}] sample {i} failed: {e}")
    if not rows:
        print(f"[{name}] no valid rows"); return
    print(f"\n===== {name}  (valid samples = {len(rows)}) =====")
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        print(f"  {k:26s} mean={v.mean():12.4f}  median={np.median(v):12.4f}"
              f"  min={v.min():12.4f}  max={v.max():12.4f}")


if __name__ == "__main__":
    for name, fn in DATASETS.items():
        try:
            run(name, fn)
        except Exception as e:
            print(f"[{name}] failed to run: {e}")
    print("\n判读：")
    print("  · compress_ratio：ASE 期望 ~0.5-1.0；CO3D 若 <<0.3 → 物体被压扁，归一化对物体失效。")
    print("  · norm_factor_mean >> median → 均值被远点拉高，换 median 可缓解。")
    print("  · CO3D 的 first_frame_sensitivity 明显大于 ASE → shuffle 第一帧使物体尺度乱抖。")
