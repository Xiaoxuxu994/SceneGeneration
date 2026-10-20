#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT 可视化 dump：把每个数据集的若干样本按"过滤前 / 诊断 / 过滤后"导出成 PLY（点云 + 相机），
用 MeshLab / CloudCompare / Open3D 直接打开，肉眼确认 gt_audit 指出的问题（98 分位截断、边缘飞点、
自由空间违例、DL3DV depth_mask、视差不足等）。只读，不改任何 loader。

和 tools/gt_audit.py 共用取样逻辑：同一 idx、同一 seed（seed·100003 + idx）=> 和体检看到的是同一组帧。

输出目录：默认 <data_root>_vis/（与数据目录并列），每个样本一个子目录：
    <data_root>_vis/<idx>_<label>_f<帧数>_<W>x<H>/
        current.ply    现在训练用的点：loader 原样输出（含 98 分位截断、z_far），RGB 着色 + 相机
        diag.ply       诊断：关掉 98 分位截断后的全部有效点，按类别着色（见 COLORS）+ 相机
        filtered.ply   建议的过滤结果：去掉 违例 / 边缘 / depth_mask 外 的点，保留被分位截断误删的好点，RGB + 相机
        cameras.ply    只有相机视锥 + 相机中心连线（黄色），看基线 / 视差最直观
        frames.png     每帧一行：RGB | 当前深度 | 诊断类别图
        stats.json     基线/深度比、各类别像素数、帧名等
    <data_root>_vis/index.csv   该数据集所有已导出样本的汇总

诊断类别（diag.ply / frames.png 的颜色，按优先级从高到低互斥）：
    违例 viol      品红  投到其他帧时跑到可见表面前面（rel < -tau），且违例次数 >= 一致次数
    depth_mask 外  青色  loader 返回了 depth_mask（如 DL3DV）但该像素在 mask 外
    边缘           红色  3x3 邻域 (max-min)/depth > 0.03
    98 分位截掉    黄色  关掉分位截断才有效、现在训练里被删掉的像素
    保留           原 RGB

用法（在 SceneGeneration 根目录）：
    python tools/gt_vis_dump.py --datasets Matterport3D DL3DV Wildrgbd --n 2 --from-audit gt_audit_out
    python tools/gt_vis_dump.py --datasets Hypersim --idx 0 61 --out-root ./gt_vis_local    # 写到本地而不是数据目录旁
不传 --datasets 时导出 config 里 weights 的全部数据集。
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
import gt_audit as G  # noqa: E402  (同目录，复用取样与几何函数；它会把工程根加入 sys.path)

COLORS = {
    "viol": (255, 0, 255),
    "mask_out": (0, 200, 255),
    "edge": (255, 40, 40),
    "pct_removed": (255, 220, 0),
}
CAM_COLOR_ENDS = np.array([[255.0, 0.0, 0.0], [0.0, 0.0, 255.0]])  # 与 vis_util.visualize_gt 一致：第一帧红 -> 最后一帧蓝
TRAJ_COLOR = (255, 200, 0)


# ---------------------------------------------------------------------------
# PLY（二进制，numpy 直接写，百万点也很快；MeshLab / CloudCompare / Open3D 均可读）
# ---------------------------------------------------------------------------
def save_ply_binary(path, points, colors):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    arr = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(arr)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def frustum_points(c2w, K, H, W, scale, color, n_per_edge=60):
    """与 vis_util.create_camera_frustum 同一几何（锥顶 + 远平面四角 8 条边），fov 取自内参，不依赖 open3d。"""
    fov_v = 2 * np.arctan(H / (2 * K[1, 1]))
    hh = np.tan(fov_v / 2) * scale
    hw = hh * (W / H)
    pc = np.array([[0, 0, 0], [-hw, -hh, scale], [hw, -hh, scale], [hw, hh, scale], [-hw, hh, scale]])
    pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    t = np.linspace(0, 1, n_per_edge)[:, None]
    pts = np.concatenate([pw[a] * (1 - t) + pw[b] * t for a, b in lines], axis=0)
    return pts, np.tile(np.asarray(color, dtype=np.float64), (len(pts), 1))


def camera_geometry(c2ws, Ks, H, W, scale):
    pts, cols = [], []
    F = len(c2ws)
    for i, (T, K) in enumerate(zip(c2ws, Ks)):
        c = CAM_COLOR_ENDS[0] * (F - i) / F + CAM_COLOR_ENDS[1] * i / F
        p, col = frustum_points(T, K, H, W, scale, c)
        pts.append(p)
        cols.append(col)
    centers = np.stack([T[:3, 3] for T in c2ws])
    t = np.linspace(0, 1, 40)[:, None]
    for a, b in zip(centers[:-1], centers[1:]):  # 相机中心按帧顺序连线：线短/重合 = 基线小
        pts.append(a * (1 - t) + b * t)
        cols.append(np.tile(TRAJ_COLOR, (len(t), 1)))
    return np.concatenate(pts), np.concatenate(cols)


# ---------------------------------------------------------------------------
# 逐像素一致性：对每帧每个像素，统计投到其他帧时一致 / 违例的次数
# ---------------------------------------------------------------------------
def per_pixel_consistency(depths, valids, Ks, c2ws, tau):
    F = len(depths)
    H, W = depths[0].shape
    rays = [G.pixel_rays(H, W, K) for K in Ks]
    w2cs = [np.linalg.inv(T) for T in c2ws]
    n_cons = [np.zeros((H, W), np.int32) for _ in range(F)]
    n_viol = [np.zeros((H, W), np.int32) for _ in range(F)]
    with np.errstate(all="ignore"):
        for i in range(F):
            vv, uu = np.nonzero(valids[i])
            if vv.size == 0:
                continue
            X = rays[i][vv, uu] * depths[i][vv, uu][:, None]
            Xw = X @ c2ws[i][:3, :3].T + c2ws[i][:3, 3]
            for j in range(F):
                if j == i:
                    continue
                Xj = Xw @ w2cs[j][:3, :3].T + w2cs[j][:3, 3]
                z = Xj[:, 2]
                front = z > 1e-6
                u = np.round(Ks[j][0, 0] * Xj[:, 0] / np.where(front, z, 1) + Ks[j][0, 2]).astype(np.int64)
                v = np.round(Ks[j][1, 1] * Xj[:, 1] / np.where(front, z, 1) + Ks[j][1, 2]).astype(np.int64)
                ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                ok[ok] = valids[j][v[ok], u[ok]]
                if not ok.any():
                    continue
                d_j = depths[j][v[ok], u[ok]]
                rel = (z[ok] - d_j) / d_j
                n_cons[i][vv[ok], uu[ok]] += (np.abs(rel) < tau)  # 同一帧对里每个源像素只出现一次，可直接加
                n_viol[i][vv[ok], uu[ok]] += (rel < -tau)
    return n_cons, n_viol


# ---------------------------------------------------------------------------
# frames.png
# ---------------------------------------------------------------------------
def depth_color(depth, valid):
    out = np.zeros(depth.shape + (3,), np.uint8)
    if not valid.any():
        return out
    d = np.log10(depth[valid] + 1e-6)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    t = 1.0 - np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)  # 近 = 暖色
    try:
        import matplotlib
        cmap = matplotlib.colormaps["turbo"] if hasattr(matplotlib, "colormaps") else matplotlib.cm.get_cmap("turbo")
        out[valid] = (np.asarray(cmap(t))[:, :3] * 255).astype(np.uint8)
    except Exception:
        out[valid] = (np.stack([t, t, t], -1) * 255).astype(np.uint8)
    return out


def save_frames_png(path, rgbs, depth_cur, valid_cur, cat_maps):
    from PIL import Image
    rows = [np.concatenate([rgb, depth_color(d, v), cm], axis=1) for rgb, d, v, cm in zip(rgbs, depth_cur, valid_cur, cat_maps)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def to_rgb(img, H, W):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:  # CHW tensor/array
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.shape[:2] != (H, W):
        from PIL import Image
        a = np.asarray(Image.fromarray(a).resize((W, H)))
    return a[..., :3]


def unproject(depth, K, c2w, mask):
    vv, uu = np.nonzero(mask)
    X = G.pixel_rays(*depth.shape, K)[vv, uu] * depth[vv, uu][:, None]
    with np.errstate(all="ignore"):
        return X @ c2w[:3, :3].T + c2w[:3, 3], vv, uu


def dump_sample(ds, z_far, name, idx, args, out_dir):
    resolution = tuple(args.res)
    seed = args.seed * 100003 + int(idx)

    G.set_percentile_threshold(True)
    views = G.fetch_views(ds, int(idx), resolution, seed)
    G.set_percentile_threshold(False)
    try:
        views_np = G.fetch_views(ds, int(idx), resolution, seed)
    finally:
        G.set_percentile_threshold(True)

    d_cur, v_cur, Ks, c2ws, dmasks = G.unpack(views, z_far)
    d_np, v_np, K_np, T_np, _ = G.unpack(views_np, z_far)
    same = all(np.allclose(a, b) for a, b in zip(c2ws, T_np)) and all(a.shape == b.shape for a, b in zip(d_cur, d_np))
    if not same:  # loader 内部用了未设种子的随机数，两次取到的帧不一致 => 退回只用当前版本
        d_np, v_np = d_cur, v_cur
    F = len(views)
    H, W = d_cur[0].shape
    rgbs = [to_rgb(v["img"], H, W) for v in views]

    # 诊断：在"关掉分位截断"的全集上算
    n_cons, n_viol = per_pixel_consistency(d_np, v_np, Ks, c2ws, args.tau)
    cats = []
    for i in range(F):
        valid = v_np[i]
        viol = valid & (n_viol[i] >= 1) & (n_viol[i] >= n_cons[i])
        mask_out = valid & ~dmasks[i] if dmasks[i] is not None else np.zeros_like(valid)
        edge = G.depth_edge_np(d_np[i], valid)
        pct = valid & ~v_cur[i] if same else np.zeros_like(valid)
        cat = np.zeros((H, W), np.uint8)  # 0 保留 1 分位截掉 2 边缘 3 mask 外 4 违例；高优先级覆盖低优先级
        cat[pct] = 1
        cat[edge] = 2
        cat[mask_out] = 3
        cat[viol] = 4
        cats.append(cat)

    med_depth = float(np.median([np.median(d[v]) for d, v in zip(d_cur, v_cur) if v.any()]))
    cam_scale = args.cam_scale if args.cam_scale > 0 else 0.08 * med_depth
    cam_pts, cam_cols = camera_geometry(c2ws, Ks, H, W, cam_scale)

    def gather(depths, masks, colorize):
        P, C = [], []
        for i in range(F):
            m = masks[i].copy()
            if args.point_stride > 1:
                keep = np.zeros_like(m)
                keep[::args.point_stride, ::args.point_stride] = True
                m &= keep
            p, vv, uu = unproject(depths[i], Ks[i], c2ws[i], m)
            P.append(p)
            C.append(colorize(i, vv, uu))
        return np.concatenate(P), np.concatenate(C)

    rgb_col = lambda i, vv, uu: rgbs[i][vv, uu]

    def diag_col(i, vv, uu):
        c = rgbs[i][vv, uu].copy()
        k = cats[i][vv, uu]
        for code, key in ((1, "pct_removed"), (2, "edge"), (3, "mask_out"), (4, "viol")):
            c[k == code] = COLORS[key]
        return c

    os.makedirs(out_dir, exist_ok=True)
    P, C = gather(d_cur, v_cur, rgb_col)
    save_ply_binary(os.path.join(out_dir, "current.ply"), np.concatenate([P, cam_pts]), np.concatenate([C, cam_cols]))
    P, C = gather(d_np, v_np, diag_col)
    save_ply_binary(os.path.join(out_dir, "diag.ply"), np.concatenate([P, cam_pts]), np.concatenate([C, cam_cols]))
    keep = [v_np[i] & (cats[i] <= 1) for i in range(F)]  # 建议过滤：去掉 违例/边缘/mask 外，保留被分位截断误删的点
    P, C = gather(d_np, keep, rgb_col)
    save_ply_binary(os.path.join(out_dir, "filtered.ply"), np.concatenate([P, cam_pts]), np.concatenate([C, cam_cols]))
    save_ply_binary(os.path.join(out_dir, "cameras.ply"), cam_pts, cam_cols)

    cat_maps = []
    for i in range(F):
        cm = (rgbs[i] * 0.35).astype(np.uint8)
        cm[~v_np[i]] = 0
        for code, key in ((1, "pct_removed"), (2, "edge"), (3, "mask_out"), (4, "viol")):
            cm[cats[i] == code] = COLORS[key]
        cat_maps.append(cm)
    save_frames_png(os.path.join(out_dir, "frames.png"), rgbs, d_cur, v_cur, cat_maps)

    centers = np.stack([T[:3, 3] for T in c2ws])
    dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)[np.triu_indices(F, 1)]
    n_valid = int(sum(v.sum() for v in v_np))
    counts = {key: int(sum((c == code).sum() for c in cats)) for code, key in
              ((1, "pct_removed"), (2, "edge"), (3, "mask_out"), (4, "viol"))}
    stats = dict(
        dataset=name, idx=int(idx), seed=seed, label=str(views[0].get("label", "")),
        frames=[str(v.get("instance", "")) for v in views], resolution=[W, H], tau=args.tau,
        percentile_compare=bool(same),
        median_depth=med_depth,
        baseline_median=float(np.median(dist)), baseline_min=float(dist.min()), baseline_max=float(dist.max()),
        baseline_over_depth=float(np.median(dist) / max(med_depth, 1e-9)),
        n_zero_baseline_pairs=int((dist < 1e-6 * max(med_depth, 1e-9)).sum()), n_pairs=int(dist.size),
        valid_pixels_no_pct=n_valid, valid_pixels_current=int(sum(v.sum() for v in v_cur)),
        **{f"n_{k}": v for k, v in counts.items()},
        **{f"frac_{k}": (v / n_valid if n_valid else float("nan")) for k, v in counts.items()},
        cam_scale=cam_scale,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


# ---------------------------------------------------------------------------
# per dataset
# ---------------------------------------------------------------------------
def pick_indices(n_total, args, name):
    idxs = list(args.idx or [])
    if args.from_audit:
        p = os.path.join(args.from_audit, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                worst = json.load(f).get("worst_samples") or []
            idxs += [int(w[1]) for w in worst[:args.n_worst]]
    if args.n > 0:
        idxs += np.linspace(0, n_total - 1, min(args.n, n_total)).round().astype(int).tolist()
    seen, out = set(), []
    for i in idxs:
        if 0 <= i < n_total and i not in seen:
            seen.add(i)
            out.append(int(i))
    return out


def vis_root_for(name, data_root, args):
    if args.out_root:
        return os.path.join(args.out_root, f"{name}_vis")
    return str(data_root).rstrip("/") + "_vis"


def run_dataset(cfg_all, name, args):
    cfg = cfg_all.train_dataset[name]
    root = cfg.get("data_root", None)
    if root is None or not G.root_exists(cfg):
        print(f"[ERROR][{name}] PATH_NOT_FOUND: data_root 不存在: {root}", flush=True)
        return
    try:
        ds, z_far = G.build_dataset(cfg_all, name, tuple(args.res), args.frames, args.mode, args.seed)
    except Exception as e:
        print(f"[ERROR][{name}] BUILD_ERROR: {type(e).__name__}: {e}", flush=True)
        return
    vis_root = vis_root_for(name, root, args)
    try:
        os.makedirs(vis_root, exist_ok=True)
    except OSError as e:
        fallback = os.path.join("gt_vis_out", f"{name}_vis")
        print(f"[WARN][{name}] 无法在数据目录旁创建 {vis_root}（{e}），改写到 {fallback}", flush=True)
        vis_root = fallback
        os.makedirs(vis_root, exist_ok=True)

    rows = []
    for idx in pick_indices(len(ds), args, name):
        t0 = time.time()
        try:
            # 临时目录带进程号：两个进程同时跑同一数据集也不会写进同一个目录
            src = os.path.join(vis_root, f".tmp_{idx:07d}_{os.getpid()}")
            tmp = dump_sample(ds, z_far, name, idx, args, src)
            label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tmp["label"])[:60]
            # 目录名带帧数和分辨率：不同参数的运行互不覆盖
            tag = f"f{args.frames}_{args.res[0]}x{args.res[1]}"
            final = os.path.join(vis_root, "_".join(x for x in (f"{idx:07d}", label, tag) if x))
            if final != src:
                if os.path.exists(final):
                    import shutil
                    shutil.rmtree(final)
                os.rename(src, final)
            rows.append(tmp)
            print(f"  [{name}] idx={idx} -> {final}  base/depth={tmp['baseline_over_depth']:.3f} "
                  f"viol={tmp['frac_viol']:.1%} edge={tmp['frac_edge']:.1%} pct={tmp['frac_pct_removed']:.1%} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [ERROR][{name}] idx={idx}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if args.verbose:
                traceback.print_exc()

    if rows:
        for r in rows:
            r["n_frames"], r["resolution"] = len(r["frames"]), f"{r['resolution'][0]}x{r['resolution'][1]}"
        keys = ["idx", "label", "n_frames", "resolution", "baseline_over_depth", "n_zero_baseline_pairs", "n_pairs", "median_depth",
                "frac_viol", "frac_mask_out", "frac_edge", "frac_pct_removed", "valid_pixels_current", "valid_pixels_no_pct"]
        p = os.path.join(vis_root, "index.csv")
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"==> [{name}] {len(rows)} samples -> {vis_root}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/datamix_all_v9-5.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=2, help="每个数据集均匀取多少个样本")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="额外指定样本下标")
    ap.add_argument("--from-audit", default=None, help="gt_audit 输出目录；取其中 worst_samples 的下标")
    ap.add_argument("--n-worst", type=int, default=2)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336], metavar=("W", "H"))
    ap.add_argument("--mode", default="train")
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0, help="与 gt_audit 相同 seed => 同一组帧")
    ap.add_argument("--point-stride", type=int, default=1, help="点云按像素下采样，文件太大时设 2")
    ap.add_argument("--cam-scale", type=float, default=0.0, help="视锥大小；0 = 自动取 0.08×深度中位数")
    ap.add_argument("--out-root", default=None, help="不写到数据目录旁，而是写到 <out-root>/<dataset>_vis")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg_all = G.load_cfg(args.config)
    names = args.datasets or list(cfg_all.train_dataset.weights.keys())
    for name in names:
        if name not in cfg_all.train_dataset:
            print(f"[ERROR][{name}] NOT_IN_CONFIG", flush=True)
            continue
        print(f"==> {name}", flush=True)
        run_dataset(cfg_all, name, args)


if __name__ == "__main__":
    main()
